"""Short CPU startup checks for RidgeFill options, using an existing tiny fixture."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    binary = args.bin_dir / 'llama-server'
    env = dict(os.environ, LD_LIBRARY_PATH=str(args.bin_dir.resolve()))
    env.pop('LLAMA_RIDGEFILL_TRIM_KV', None)
    help_run = subprocess.run([str(binary), '--help'], capture_output=True, text=True, env=env, check=True)
    help_text = help_run.stdout + help_run.stderr
    (args.out / 'help.txt').write_text(help_text)
    assert '--ridgefill-trim-kv' in help_text and 'reuse S1 bulk K/V in the late sweep (default: on)' in help_text
    checks = ['help']
    if args.coexistence:
        for parallel in (1, 2):
            for label, flags, named in [('draft', ['-md', 'missing-draft.gguf'], '--model-draft'),
                                        ('mtp', ['--spec-type', 'draft-mtp'], 'built-in MTP')]:
                command = [str(binary), '-m', 'missing-model.gguf', '--ridgefill', 'missing-projector.gguf', '-np', str(parallel), *flags]
                rejected = subprocess.run(command, capture_output=True, text=True, env=env, timeout=15)
                text = rejected.stdout + rejected.stderr
                (args.out / (label + '-' + str(parallel) + '.log')).write_text(text)
                assert rejected.returncode > 0 and named in text and ('remove --' in text or '--spec-type none' in text)
                assert 'GGML_ASSERT' not in text and 'loading model' not in text and 'failed to load model' not in text
                checks.append(label + ':' + str(parallel))
        (args.out / 'result.json').write_text(json.dumps(dict(result='PASS', checks=checks), indent=2) + '\n')
        return
    if args.retired_certificate:
        assert '--ridgefill-certificate' not in help_text
        for name in ('llama-server', 'llama-ridgefill-score'):
            exe = args.bin_dir / name
            help_run = subprocess.run([str(exe), '--help'], capture_output=True, text=True, env=env, check=True)
            assert '--ridgefill-certificate' not in help_run.stdout + help_run.stderr
            for flag in ('--ridgefill-certificate', '--ridgefill-certificate-tau', '--ridgefill-certificate-cap'):
                invalid = subprocess.run([str(exe), flag, 'on' if flag.endswith('certificate') else '0.1'], capture_output=True, text=True, env=env)
                text = invalid.stdout + invalid.stderr
                assert invalid.returncode > 0 and flag in text and 'GGML_ASSERT' not in text
                checks.append(name + ':' + flag)
        (args.out / 'result.json').write_text(json.dumps(dict(result='PASS', checks=checks), indent=2) + '\n')
        return
    bad = subprocess.run([str(binary), '--ridgefill-trim-kv', 'invalid'], capture_output=True, text=True, env=env)
    assert bad.returncode > 0 and '--ridgefill-trim-kv must be on or off' in bad.stderr and 'GGML_ASSERT' not in bad.stderr
    checks.append('invalid-value')
    for label, flags, override, expected in [('default', [], None, 'on'), ('on', ['on'], None, 'on'),
                                               ('off', ['off'], None, 'off'), ('legacy', ['off'], '1', 'on')]:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        command = [str(binary), '-m', str(args.model), '--ridgefill', str(args.projector), '--ridgefill-mode', 'quality+qattn',
                   '--ridgefill-alpha', '0', '--ridgefill-s1-window', '64', '-c', '2304', '-b', '128', '-ub', '128',
                   '-np', '1', '-t', '4', '-ngl', '0', '-fa', 'off', '--no-warmup', '--port', str(port)]
        if flags:
            command += ['--ridgefill-trim-kv', *flags]
        case_env = dict(env)
        if override is not None:
            case_env['LLAMA_RIDGEFILL_TRIM_KV'] = override
        with (args.out / (label + '.log')).open('w') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=case_env)
            try:
                deadline = time.monotonic() + 60
                while True:
                    assert proc.poll() is None, label + ': server exited'
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:' + str(port) + '/health', timeout=1):
                            break
                    except OSError:
                        assert time.monotonic() < deadline, label + ': startup timeout'
                        time.sleep(.05)
            finally:
                proc.terminate()
                proc.wait(timeout=10)
        text = (args.out / (label + '.log')).read_text()
        assert 'RidgeFill: trim K/V ' + expected in text and 'GGML_ASSERT' not in text, label
        if override is not None:
            assert 'deprecated and overrides --ridgefill-trim-kv' in text
        checks.append(label)
    (args.out / 'result.json').write_text(json.dumps(dict(result='PASS', checks=checks), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--projector', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--retired-certificate', action='store_true')
    parser.add_argument('--coexistence', action='store_true')
    run(parser.parse_args())
