"""Compare teacher-forced server logprobs with direct F32 logits and check eval controls."""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--engine-dir", type=Path, required=True, help="directory containing rf_dump and build/bin/llama-server")
parser.add_argument("--fixture", type=Path, default=root / "out/gate1-qwen35")
parser.add_argument("--work", type=Path, default=root / "out/eval-client-control")
parser.add_argument("--port", type=int, default=18087)
parser.add_argument("--kva", type=Path, required=True)
args = parser.parse_args()
args.work.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / "tests/ridgefill"))
from eval_client import Client, EvalError, completion_body, paired_summary, score_nll, target_logprob

directory = args.engine_dir.resolve()
model = args.fixture.resolve() / "tiny-qwen35-f32.gguf"
tokens = args.fixture.resolve() / "tokens.bin"
prefix = args.work.resolve() / "native-control"
subprocess.run([str(directory / "rf_dump"), "--model", str(model), "--tokens", str(tokens), "--output", str(prefix),
                "--tail", "700", "--keep", "513", "--batch", "700", "--ubatch", "128"],
               stdout=(args.work / "native-control.log").open("w"), stderr=subprocess.STDOUT, check=True)
ids = np.fromfile(tokens, dtype=np.int32).tolist()
logits = np.fromfile(str(prefix) + ".logits", dtype=np.float32).reshape(513, 300)[:-1].astype(np.float64)
shift = logits.max(axis=1)
normalizer = shift + np.log(np.exp(logits - shift[:, None]).sum(axis=1))
expected = logits[np.arange(512), ids[-512:]] - normalizer

with (args.work / "stock-server.log").open("w") as log:
    server = subprocess.Popen([str(directory / "build/bin/llama-server"), "-m", str(model), "-ngl", "0", "-c", "1024",
                               "-b", "700", "-ub", "128", "-np", "1", "-t", "4", "--host", "127.0.0.1", "--port", str(args.port),
                               "--no-warmup", "--offline", "-ctk", "f32", "-ctv", "f32", "-fa", "off"], stdout=log, stderr=subprocess.STDOUT)
    client = Client(f"http://127.0.0.1:{args.port}", timeout=30)
    try:
        deadline = time.monotonic() + 30
        while True:
            if server.poll() is not None:
                raise RuntimeError("stock server exited, see out/eval-stock-server.log")
            try:
                client.request("/health")
                break
            except (urllib.error.URLError, EvalError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        a = score_nll(client, ids, "off", 600, 1, 1)
        b = score_nll(client, ids, "off", 600, 1, 1)
        np.testing.assert_allclose(a["logprobs"], expected, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(a["logprobs"], b["logprobs"])
        response, first, wall = client.ttft(completion_body(ids, "off", 600))
        assert first > 0 and wall >= first and response["timings"]["predicted_n"] == 1
        body = completion_body(ids[:188], "off", 600)
        body.update(n_probs=1, post_sampling_probs=True, logit_bias=[[ids[188], 1000000.0]])
        post_sampling = client.request("/completion", body)
        try:
            target_logprob(post_sampling, ids[188])
        except EvalError:
            pass
        else:
            raise AssertionError("post-sampling control was accepted as raw logprob")
        controls: list[dict] = [dict(name=str(i), reference=dict(nll=1+i), candidate=dict(nll=1+i+delta)) for i, delta in enumerate([0.01, 0.02, 0.03, 0.04])]
        positive = paired_summary(controls, "nll", args.kva)
        assert positive["ci95"][0] > 0
        for row in controls:
            row["candidate"] = dict(row["reference"])
        negative = paired_summary(controls, "nll", args.kva)
        assert negative["delta"] == 0 and negative["degenerate"]
        print(json.dumps(dict(result="PASS", bca_positive_control=positive["delta"], bca_negative_control=negative["delta"], scope="stock eval-client controls only", tokens=512,
                              nll=a["nll"], max_logprob_abs_error=float(np.max(np.abs(np.array(a["logprobs"]) - expected))),
                              negative_off_identical=True, post_sampling_control="REFUSED", ttft_s=first, wall_s=wall)))
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
