#!/usr/bin/env python3
"""Check the production lexical selector against Q6's 68 frozen fixtures and byte-token controls."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata

FIELDS = ('window', 'rows', 'matches', 'spans', 'candidates', 'capped', 'clipped', 'oversize_skipped', 'row_cap', 'anchors')
EXPECTED_SHA256 = '7563729fcc715f05c2c750f98271ef12007fde40ac646bfa17a73a7fe6b58469'


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    expected_path = Path(__file__).with_name('lexical-v2-expected.jsonl')
    assert hashlib.sha256(expected_path.read_bytes()).hexdigest() == EXPECTED_SHA256

    def select(model, source, label):
        output = args.out / (label + '.jsonl')
        with (args.out / (label + '.log')).open('w') as log:
            subprocess.run([str(args.binary), str(model), str(source), str(output)], stdout=log, stderr=log, check=True, timeout=120)
        return {r['id']: r['lexical'] for r in map(json.loads, output.read_text().splitlines())}

    observed = select(args.model, args.tokenized, 'frozen')
    expected = {r['id']: r['lexical'] for r in map(json.loads, expected_path.read_text().splitlines())}
    assert len(expected) == len(observed) == 68
    for key, value in expected.items():
        for field in FIELDS:
            assert observed[key][field] == value[field], (key, field, observed[key][field], value[field])
    sources = [json.loads(line) for line in args.tokenized.read_text().splitlines()]
    for source in sources:
        result = observed[source['id']]
        assert result['window'][1] <= source['selector_end']
        assert all(not source['excluded'][row] for row in result['rows'])

    cases = []
    def synthetic(label, prefix, question, padding=400, params=None):
        body = (('neutral filler.\n' * padding).encode() + (prefix.encode() if isinstance(prefix, str) else prefix))
        query = question.encode() if isinstance(question, str) else question
        raw = body + b' ' * (4096 - len(query)) + query
        case = dict(id=label, ids=[3 + c for c in raw], n_prefix=len(body), params=params or {})
        cases.append(case)
        return len(body) - len(prefix.encode() if isinstance(prefix, str) else prefix)

    line = 'One of the special magic UUIDS for amber-finch is: 1234.\n'
    query = 'the special magic uuid for amber finch'
    start = synthetic('positive', line, query)
    synthetic('two', line*2, query)
    synthetic('three', line*3, query)
    synthetic('rarity-one', line*2, query, params=dict(maximum_occurrences=1))
    synthetic('rarity-three', line*3, query, params=dict(maximum_occurrences=3))
    synthetic('negative', line, 'completely different request about oceans')
    synthetic('newline', 'alpha beta.\ngamma delta.\n', 'alpha beta gamma delta')
    synthetic('nul', 'alpha beta\0gamma delta.\n', 'alpha beta gamma delta')
    synthetic('full-window', 'alpha beta gamma delta.\n', 'alpha beta gamma delta')
    synthetic('cut-window', 'alpha beta gamma delta.\n', 'alpha beta gamma delta', params=dict(window_tokens=20))
    unicode_line = 'alpha b\u00e9ta gamma delta.\n'
    unicode_start = synthetic('unicode', unicode_line, 'alpha b\u00e9ta gamma delta')
    synthetic('cap', line, query, padding=30)
    synthetic('clipped', 'alpha beta gamma delta' + ' example'*30 + '.\n', 'alpha beta gamma delta')
    synthetic('late', 'intro '*30 + 'alpha beta gamma delta.\n', 'alpha beta gamma delta')
    synthetic('span-20', 'alpha beta gamma delta.\n', 'alpha beta gamma delta', params=dict(span_rows=20))
    synthetic('min-5', 'alpha beta gamma delta.\n', 'alpha beta gamma delta', params=dict(minimum_words=5))
    synthetic('min-2', 'alpha beta.\n', 'alpha beta', params=dict(minimum_words=2))
    synthetic('underscores', 'alpha_beta-gamma\'delta.\n', 'alpha beta gamma delta')
    synthetic('ascii-fold', 'ALPHA BETA GAMMA DELTA.\n', 'alpha beta gamma delta')
    synthetic('nonascii-case', 'alpha B\u00c9TA gamma delta.\n', 'alpha b\u00e9ta gamma delta')
    synthetic('combining', 'alpha be\u0301ta gamma delta.\n', 'alpha b\u00e9ta gamma delta')
    synthetic('decimal', 'alpha 12.34 beta gamma.\n', 'alpha 12 34 beta gamma')
    synthetic('cjk-boundary', 'alpha beta\u3002gamma delta.\n', 'alpha beta gamma delta')
    synthetic('invalid-utf8', b'alpha be\xffta gamma delta.\n', 'alpha beta gamma delta')
    synthetic('unicode16', 'alpha \U000105c0 beta gamma.\n', 'alpha \U000105c0 beta gamma')
    synthetic('whitespace-boundary', 'alpha beta.\u001cgamma delta.\n', 'alpha beta gamma delta')
    # A complete rare phrase is insufficient when every unigram occurs at least three times.
    synthetic('frequent-words', 'alpha beta gamma delta.\nalpha beta.\nalpha beta.\ngamma delta.\ngamma delta.\n', 'alpha beta gamma delta')
    source_path = args.out / 'controls-input.jsonl'
    source_path.write_text(''.join(json.dumps(case) + '\n' for case in cases))
    controls = select(args.tiny_model, source_path, 'controls')
    assert controls['positive']['spans'] == [[start, start + len(line) - 1]]
    assert len(controls['two']['spans']) == 2 and len(controls['rarity-three']['spans']) == 3
    for key in ('three', 'rarity-one', 'negative', 'newline', 'nul', 'cut-window', 'late', 'span-20', 'min-5',
                'nonascii-case', 'combining', 'cjk-boundary', 'invalid-utf8', 'whitespace-boundary', 'frequent-words'):
        assert controls[key]['rows'] == [], key
    for key in ('full-window', 'unicode', 'clipped', 'min-2', 'underscores', 'ascii-fold', 'decimal', 'unicode16'):
        assert controls[key]['rows'], key
    assert controls['unicode']['spans'] == [[unicode_start, unicode_start + len(unicode_line.encode()) - 1]]
    assert controls['cap']['capped'] and not controls['cap']['rows']
    assert controls['clipped']['clipped'] and all(b-a <= 80 for a, b in controls['clipped']['spans'])
    assert controls['late']['oversize_skipped'] > 0
    assert controls['span-20']['clipped'] > 0 and controls['span-20']['oversize_skipped'] > 0
    assert all(1 <= a['count'] <= 2 for item in controls['positive']['anchors'] for a in item['anchors'])

    # Check the tokenizer table plus the selector's local additions against the frozen Unicode version.
    assert unicodedata.unidata_version == '16.0.0'
    root = Path(__file__).resolve().parents[2]
    table = (root/'src/unicode-data.cpp').read_text().split('unicode_ranges_flags = {', 1)[1].split('};', 1)[0]
    ranges = [(int(a, 16), int(b, 16)) for a, b in re.findall(r'\{0x([0-9a-fA-F]+), 0x([0-9a-fA-F]+)\}', table)]
    table = (root/'common/common.cpp').read_text().split('static const std::array<uint32_t, 2> added[] = {', 1)[1].split('};', 1)[0]
    added = [(int(a, 16), int(b, 16)) for a, b in re.findall(r'\{0x([0-9a-fA-F]+), 0x([0-9a-fA-F]+)\}', table)]
    index = supplement = 0
    for c in range(0x110000):
        while index+1 < len(ranges) and ranges[index+1][0] <= c:
            index += 1
        while supplement < len(added) and added[supplement][1] < c:
            supplement += 1
        word = bool(ranges[index][1] & 6) or (supplement < len(added) and added[supplement][0] <= c)
        assert word == (unicodedata.category(chr(c))[0] in 'LN'), hex(c)
    summary = dict(result='PASS', frozen=68, controls=len(cases), suffix_invariance=68+len(cases),
                   unicode_codepoints=0x110000, fixture_sha256=EXPECTED_SHA256,
                   tokenized_sha256=hashlib.sha256(args.tokenized.read_bytes()).hexdigest())
    (args.out/'results.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'model', 'tiny-model', 'tokenized', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
