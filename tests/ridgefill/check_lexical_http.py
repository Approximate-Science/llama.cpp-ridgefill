#!/usr/bin/env python3
"""Short tiny-model lexical v2 native/HTTP option, report and no-op controls."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    ids = [3 + ord(' ')] * 2000
    line = 'One of the special magic UUIDS for amber-finch is: 1234.\n'
    for pos, text in ((100, line), (300, line), (500, line.replace('amber-finch', 'other-bird')),
                      (700, line.replace('amber-finch', 'third-bird')), (1940, 'the special magic uuid for amber finch')):
        ids[pos:pos + len(text)] = [3 + ord(c) for c in text]
    source = args.out/'native-input.json'
    source.write_text(json.dumps(dict(ids=[*ids, ids[-1]]))+'\n')
    url = 'http://127.0.0.1:' + str(args.port)
    def request(path, body=None):
        req = urllib.request.Request(url+path, data=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)

    command = [args.bin_dir/'llama-server', '-m', args.model, '--ridgefill', args.projector,
               '--ridgefill-mode', 'quality+qattn', '--ridgefill-share', '0', '--ridgefill-digits', 'off', '--ridgefill-alpha', '0',
               '--ridgefill-tail', '400', '--ridgefill-min-prompt', '1', '--ridgefill-qattn-share', '.2',
               '--ridgefill-s1-window', '64', '-b', '128', '-ub', '128', '-c', '2304', '-np', '1', '-t', '4',
               '-ngl', '0', '-ctk', 'f32', '-ctv', 'f32', '-fa', 'off', '--cache-ram', '0', '--no-warmup',
               '--host', '127.0.0.1', '--port', str(args.port), '--ridgefill-lexical-window', '128',
               '--ridgefill-lexical-min-words', '3', '--ridgefill-lexical-span-rows', '79', '--ridgefill-lexical-max-occurrences', '3']
    checks = []
    def check(ok, label):
        assert ok, label
        checks.append(label)

    with (args.out/'server.log').open('w') as log:
        proc = subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(1200):
                assert proc.poll() is None, 'server exited'
                try:
                    request('/health')
                    break
                except urllib.error.URLError:
                    time.sleep(.05)
            else:
                raise RuntimeError('startup timeout')
            cases = [('off', dict(ridgefill_lexical=False)), ('inherited', dict(ridgefill_lexical=True)),
                     ('default', dict(ridgefill_lexical=True, ridgefill_lexical_window=64, ridgefill_lexical_min_words=4,
                                      ridgefill_lexical_span_rows=80, ridgefill_lexical_max_occurrences=2)),
                     ('window', dict(ridgefill_lexical_window=20)), ('minimum', dict(ridgefill_lexical_min_words=8)),
                     ('span', dict(ridgefill_lexical_span_rows=20)), ('rarity', dict(ridgefill_lexical_max_occurrences=1)),
                     ('off-again', dict(ridgefill_lexical=False))]
            responses = {}
            defaults = dict(window=64, min_words=4, span_rows=80, max_occurrences=2)
            for label, extra in cases:
                if label not in ('off', 'inherited', 'off-again'):
                    extra = {**dict(ridgefill_lexical=True), **{'ridgefill_lexical_'+k: v for k,v in defaults.items()}, **extra}
                response = request('/completion', dict(prompt=ids, n_predict=1, n_probs=300, temperature=0, seed=123,
                                                       ignore_eos=True, cache_prompt=False, return_tokens=True, **extra))
                (args.out/(label+'-http.json')).write_text(json.dumps(response)+'\n')
                responses[label] = response
                knobs = dict(window=128, min_words=3, span_rows=79, max_occurrences=3) if label == 'inherited' else defaults
                knobs = {k: extra.get('ridgefill_lexical_'+k, v) for k,v in knobs.items()}
                output = args.out/(label+'-native.jsonl')
                native = [args.bin_dir/'llama-ridgefill-score', '--model', args.model, '--projector', args.projector,
                          '--input', source, '--output', output, '--scored', '1', '--tail', '401', '--min-prompt', '1',
                          '--mode', 'quality+qattn', '--share', '0', '--digits', 'off', '--qattn-share', '.2', '--alpha', '0',
                          '--ridgefill-s1-window', '64', '--qattn-window-end', 'known', '--batch', '128', '--ubatch', '128',
                          '--threads', '4', '--cache-type', 'f32', '--h-type', 'f16', '--flash-attn', 'off', '--diagnostics',
                          '--ridgefill-lexical', 'on' if extra.get('ridgefill_lexical', True) else 'off']
                for k,v in knobs.items():
                    native += ['--ridgefill-lexical-'+k.replace('_','-'), str(v)]
                with (args.out/(label+'-native.log')).open('w') as native_log:
                    subprocess.run(list(map(str, native)), stdout=native_log, stderr=native_log, check=True, timeout=120)
                expected = json.loads(output.read_text())
                report = response['timings']['ridgefill']
                check(report['n_exact_prefix'] == expected['ridgefill']['n_exact_prefix'], label+'-budget')
                check(report['n_exact_prefix_by_source'] == expected['ridgefill']['n_exact_prefix_by_source'], label+'-sources')
                if extra.get('ridgefill_lexical', True):
                    check(report['lexical'] == expected['ridgefill']['lexical'], label+'-metadata-parity')
                    lexical = report['lexical']
                    check([lexical[k] for k in ('window_tokens','minimum_words','span_rows','maximum_occurrences')] == list(knobs.values()), label+'-resolved-parameters')
                    check(report['n_exact_prefix_by_source']['lexical'] == lexical['n_rows'], label+'-exact-lexical-attribution')
                    check(set(lexical['rows']) <= set(expected['exact_rows']), label+'-mandatory-union')
                    check(lexical['n_rows'] <= report['n_prefix']//20, label+'-prefix-cap')
                else:
                    check('lexical' not in report, label+'-off-report')
            net = responses['default']['timings']['ridgefill']['lexical']
            check(net['n_rows'] == 56 and net['capped'] and net['n_matches'] == 1, 'whole-span-cap-positive')
            for label in ('window', 'minimum', 'span', 'rarity'):
                check(responses[label]['timings']['ridgefill']['lexical']['n_rows'] == 0, label+'-negative')
            check(responses['span']['timings']['ridgefill']['lexical']['oversize_skipped'] > 0, 'oversize-visible')
            payload = lambda r: {k:r[k] for k in ('tokens','content','completion_probabilities')}
            for label in ('window', 'minimum', 'span', 'rarity', 'off-again'):
                check(payload(responses['off']) == payload(responses[label]), label+'-no-op-byte-identity')
            for key, low in (('window',1),('min_words',2),('span_rows',1),('max_occurrences',1)):
                for value in (low-1, 4097, 2.5, '3', True):
                    try:
                        request('/completion', dict(prompt=ids, n_predict=1, **{'ridgefill_lexical_'+key:value}))
                    except urllib.error.HTTPError as error:
                        check(error.code == 400 and 'integer in' in error.read().decode(), 'reject-'+key+'-'+str(value))
                    else:
                        raise AssertionError('accepted invalid lexical parameter')
        finally:
            proc.terminate()
            proc.wait(timeout=15)
    result = dict(result='PASS', checks=len(checks), records=checks)
    (args.out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(result='PASS', checks=len(checks))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bin-dir','model','projector','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--port', type=int, default=18316)
    run(parser.parse_args())
