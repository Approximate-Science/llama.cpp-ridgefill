#!/usr/bin/env python3
"""G5 functional parity (L-A): the archive build and the dev build answer the same greedy requests (mode off, speed,
quality class+digit at alpha 0) with identical tokens and top-5 ids and |delta logprob| <= 1e-5 (byte identity only
holds for identical build flags: GGML_NATIVE on/off rounds differently, which mode off, stock code, shows as well),
and the fill is active (500 of 700 rows) where it should be. Usage: g5_parity.py BIN_A BIN_B QWEN35_FIXTURE_DIR OUT_DIR"""
import json, subprocess, sys, time, urllib.request
from pathlib import Path
if len(sys.argv) < 5:
    sys.exit(__doc__)
A, B, Q35, OUT = (Path(a).resolve() for a in sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)
raw = (Q35 / "tokens.bin").read_bytes()
TOK = [int.from_bytes(raw[i:i + 4], "little", signed=True) for i in range(0, len(raw), 4)]
REQ = [("off", dict(ridgefill=False)), ("speed", dict()), ("quality", dict(ridgefill_mode="quality"))]

def serve(bin_dir, port, tag):
    log = open(OUT / f"server-{tag}.txt", "w")
    p = subprocess.Popen([str(bin_dir / "llama-server"), "-m", str(Q35 / "tiny-qwen35-f32.gguf"), "--ridgefill", str(Q35 / "projector-s3-f32.gguf"),
                          "--ridgefill-tail", "200", "--ridgefill-min-prompt", "1", "--ridgefill-alpha", "0", "-ngl", "0", "-t", "4",
                          "--no-warmup", "-c", "1024", "-b", "700", "-ub", "128", "-np", "1", "-ctk", "f32", "-ctv", "f32", "-fa", "off",
                          "--host", "127.0.0.1", "--port", str(port)], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200:
                break
        except Exception:
            time.sleep(0.1)
    out = {}
    for name, extra in REQ:
        body = dict(prompt=TOK, n_predict=8, temperature=0, n_probs=5, cache_prompt=False, **extra)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=300).read())
        out[name] = dict(tokens=r.get("tokens"), content=r.get("content"), probs=r.get("completion_probabilities"),
                         filled=r.get("timings", {}).get("prompt_filled_n", 0))
    p.terminate(); p.wait(timeout=30); log.close()
    (OUT / f"responses-{tag}.json").write_text(json.dumps(out, indent=1))
    return out

ra, rb = serve(A, 18751, "a"), serve(B, 18752, "b")
records = []
def compare(x, y):
    worst, ids = 0.0, True
    for pa, pb in zip(x["probs"], y["probs"]):
        la, lb = pa.get("top_logprobs", []), pb.get("top_logprobs", [])
        ids &= [t["id"] for t in la] == [t["id"] for t in lb] and len(la) == 5
        worst = max([worst, abs(pa["logprob"] - pb["logprob"])] + [abs(u["logprob"] - v["logprob"]) for u, v in zip(la, lb)])
    # "tokens" is empty without return_tokens; the chosen ids per step and the text carry the generated tokens
    # (probs merge partial-UTF-8 byte tokens, so there can be fewer than n_predict entries)
    chosen = [p["id"] for p in x["probs"]] == [p["id"] for p in y["probs"]] and len(x["probs"]) == len(y["probs"]) > 0
    return chosen and x["content"] == y["content"] and len(x["content"]) > 0, ids, worst

for name, _ in REQ:
    same = json.dumps(ra[name], sort_keys=True) == json.dumps(rb[name], sort_keys=True)
    toks, ids, worst = compare(ra[name], rb[name])
    want = 0 if name == "off" else 500
    ok = toks and ids and worst <= 1e-5 and ra[name]["filled"] == rb[name]["filled"] == want
    records.append(dict(check=f"parity-{name}", result="PASS" if ok else "FAIL", byte_identical=same, chosen_and_text_equal=toks,
                        top5_ids_equal=ids, max_abs_dlogprob=worst, filled=ra[name]["filled"], expect_filled=want))
    print(json.dumps(records[-1]))
diff_modes = ra["speed"]["probs"] != ra["off"]["probs"] and ra["quality"]["probs"] != ra["speed"]["probs"]
records.append(dict(check="control-modes-differ", result="PASS" if diff_modes else "FAIL"))
print(json.dumps(records[-1]))
s = dict(gate="G5-parity (L-A)", checks=len(records), passed=sum(r["result"] == "PASS" for r in records))
s["result"] = "PASS" if s["passed"] == s["checks"] else "FAIL"
(OUT / "results.json").write_text(json.dumps(dict(summary=s, checks=records, a=str(A), b=str(B)), indent=2) + "\n")
print(json.dumps(s)); sys.exit(0 if s["result"] == "PASS" else 1)
