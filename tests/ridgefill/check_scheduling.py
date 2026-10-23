#!/usr/bin/env python3
"""Check sweep progress beside two HTTP streams; record stream gaps and optional VRAM use.

The JSON config supplies the server command, prompt/background token arrays, logical
batch size, and optional mode/port/environment/VRAM counter. No model is downloaded.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request


def run(config, out):
    out.mkdir(parents=True, exist_ok=False)
    port = config.get("port", 18189)
    url = f"http://127.0.0.1:{port}"
    command = [str(value) for value in config["command"]] + ["--host", "127.0.0.1", "--port", str(port)]
    environment = os.environ.copy()
    environment.update(config.get("env", {}))
    started = [threading.Event(), threading.Event()]
    stop_streams = threading.Event()
    times = [[], []]
    chunks = [[], []]
    errors = []

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=config.get("timeout", 180))

    def stream(index):
        body = dict(prompt=config["background"], n_predict=config.get("generate", 512), temperature=0,
                    stream=True, ignore_eos=True, cache_prompt=False, ridgefill_mode="off", id_slot=index,
                    logit_bias=[[config.get("target", 7), 1000000]])
        try:
            with request("/completion", body) as response:
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    if line.strip() == b"data: [DONE]":
                        break
                    chunk = json.loads(line[6:])
                    chunks[index].append(chunk)
                    assert not chunk.get("error"), chunk
                    if chunk.get("stop"):
                        continue
                    times[index].append(time.monotonic())
                    if stop_streams.is_set():
                        break
                    if len(times[index]) >= 3:
                        started[index].set()
        except Exception as error:
            errors.append(repr(error))
            started[index].set()
            raise

    vram = Path(config["vram"]) if config.get("vram") else None
    baseline = int(vram.read_text()) if vram else None
    peak = [baseline]
    stop_monitor = threading.Event()

    def monitor():
        while not stop_monitor.wait(0.05):
            peak[0] = max(peak[0], int(vram.read_text()))

    with (out / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        thread = threading.Thread(target=monitor, daemon=True) if vram else None
        if thread:
            thread.start()
        try:
            deadline = time.monotonic() + 120
            while True:
                assert server.poll() is None, "server exited; see server.log"
                try:
                    with request("/health"):
                        pass
                    assert server.poll() is None, "server failed to bind; see server.log"
                    break
                except urllib.error.URLError:
                    assert time.monotonic() < deadline, "startup timeout"
                    time.sleep(0.05)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(stream, index) for index in range(2)]
                try:
                    for event in started:
                        assert event.wait(60), "generator did not start"
                    assert not errors, errors
                    begin = time.monotonic()
                    body = dict(prompt=config["prompt"], n_predict=1, temperature=0, cache_prompt=False,
                                n_probs=config.get("n_probs", 1), id_slot=2, ignore_eos=True,
                                ridgefill_mode=config.get("mode", "quality+qattn"))
                    try:
                        with request("/completion", body) as response:
                            completion = json.load(response)
                    except urllib.error.HTTPError as error:
                        assert config.get("expect_error"), str(error)
                        completion = json.load(error)
                    end = time.monotonic()
                    if config.get("expect_error"):
                        assert config["expect_error"] in json.dumps(completion), completion
                    else:
                        assert "error" not in completion, completion
                    assert all(not future.done() for future in futures), "generators finished before the prompt"
                finally:
                    stop_streams.set()
                for future in futures:
                    future.result(timeout=180)
        finally:
            stop_monitor.set()
            if thread:
                thread.join()
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

    (out / "response.json").write_text(json.dumps(completion, indent=2) + "\n")
    (out / "streams.json").write_text(json.dumps(dict(times=times, chunks=chunks, begin=begin, end=end)) + "\n")
    text = (out / "server.log").read_text(errors="replace")
    fields = ("iteration", "phase", "cursor", "next", "cursor_next", "rows", "charged", "remaining")
    pattern = (r"RidgeFill step iteration=(\d+) phase=(\w+) cursor=(\d+) next=(\w+) "
               r"cursor_next=(\d+) rows=(\d+) charged=(\d+) remaining=(\d+)")
    steps = [dict(zip(fields, match)) for match in re.findall(pattern, text)]
    for step in steps:
        for field in fields:
            if field not in ("phase", "next"):
                step[field] = int(step[field])
    if config.get("expect_steps", True):
        assert steps, "no sweep steps in trace (enable --verbose)"
        iterations = [step["iteration"] for step in steps]
        assert sorted(set(iterations)) == list(range(iterations[0], iterations[-1] + 1)), iterations
        groups = [[step for step in steps if step["iteration"] == iteration] for iteration in sorted(set(iterations))]
        assert all(sum(step["charged"] for step in group) <= config["batch"] - 2 for group in groups)
        if config.get("expect_multi_step"):
            assert any(len(group) > 1 for group in groups), "unused work budget: only one step per iteration"
            assert any(sum(step["rows"] for step in group) > config["batch"] for group in groups), "cheap rows were not charged by cost"
        assert all(step["rows"] <= config["batch"] - 2 and step["charged"] <= config["batch"] - 2 for step in steps)
        assert all(step["next"] != step["phase"] or step["cursor_next"] > step["cursor"]
                   for step in steps if step["phase"] != "external")
    gaps = []
    for stream_times in times:
        overlap = [right - left for left, right in zip(stream_times, stream_times[1:]) if right >= begin and left <= end]
        assert overlap, "no concurrent stream gaps measured"
        gaps.append(dict(n=len(overlap), max_s=max(overlap), median_s=statistics.median(overlap)))
    result = dict(result="PASS", command=command, env=config.get("env", {}), wall_s=end - begin,
                  binary_sha256=hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(), steps=steps, gaps=gaps,
                  baseline_bytes=baseline, peak_bytes=peak[0], compute=re.findall(r".*compute buffer.*", text),
                  generated=[len(stream_times) for stream_times in times], target_prompt_n=len(config["prompt"]))
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text()), args.out)), flush=True)
