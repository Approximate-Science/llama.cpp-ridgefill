"""Evaluation controls: fixed fill boundary, exact target count, no answer suffix, and named refusals."""

import copy
import math
import unittest

import eval_client as evaluation


class FakeClient(evaluation.Client):
    def __init__(self):
        self.calls = []

    def request(self, path: str, body: dict | None = None) -> dict:
        if body is None:
            raise AssertionError("this control expects POST requests")
        self.calls.append((path, body))
        if path == "/completion":
            target = body["logit_bias"][0][0]
            normalizer = math.log(sum(math.exp(-i / 10) for i in range(17)))
            n = len(body["prompt"])
            prefix = n - body["ridgefill_tail"] if body["ridgefill"] else 0
            report = dict(n_prompt=n, n_prefix=prefix, n_filled=prefix, n_exact_prefix=0, n_cached_exact_prefix=0, n_exact_tail=n-prefix,
                          n_exact_prefix_by_source=dict.fromkeys(("class", "digit", "special", "header", "qattn"), 0),
                          tokens_skipped=prefix/n, macs_exact=100.0, macs_ridgefill=80.0 if prefix else 100.0,
                          mac_share_skipped=0.2 if prefix else 0.0)
            return dict(completion_probabilities=[dict(id=target, logprob=-target / 10 - normalizer)],
                        timings=dict(prompt_filled_n=100 if not body["cache_prompt"] and body["ridgefill"] else 0, ridgefill=report))
        if path == "/detokenize":
            return dict(content="haystack and question")
        if path == "/apply-template":
            return dict(prompt="<|im_start|>user\nhaystack and question<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        raise AssertionError(path)


class EvaluationTests(unittest.TestCase):
    def test_tail_512_and_fixed_prefix(self):
        ids = [i % 17 for i in range(700)]
        client = FakeClient()
        result = evaluation.score_nll(client, ids, "quality", tail=600, min_prompt=1, tail_min=1)
        self.assertEqual(result["scored"], 512)
        self.assertEqual(len(client.calls), 512)
        self.assertEqual(result["prefix"], 100)
        normalizer = math.log(sum(math.exp(-i / 10) for i in range(17)))
        expected = sum(token / 10 + normalizer for token in ids[-512:]) / 512
        self.assertAlmostEqual(result["nll"], expected)
        for index, (path, body) in enumerate(client.calls):
            self.assertEqual(path, "/completion")
            self.assertEqual(len(body["prompt"]) - body["ridgefill_tail"], 100)
            self.assertGreaterEqual(len(body["prompt"]) - 1, 100)
            self.assertEqual(body["logit_bias"][0][0], ids[188 + index])
            self.assertFalse(body["post_sampling_probs"])
        self.assertFalse(client.calls[0][1]["cache_prompt"])
        self.assertTrue(client.calls[-1][1]["cache_prompt"])

    def test_reporting_is_required_and_checked(self):
        with self.assertRaisesRegex(evaluation.EvalError, "missing RidgeFill reporting"):
            evaluation.validate_ridgefill_report(None, 700, "quality")
        stock = evaluation.response_ridgefill_report(dict(timings={}), 700, "off")
        self.assertEqual(stock["mac_share_skipped"], 0)
        self.assertIsNone(stock["macs_exact"])
        client = FakeClient()
        body = evaluation.completion_body([1] * 700, "speed", 600)
        body["logit_bias"] = [[1, 1000000.0]]
        row = client.request("/completion", body)
        report = row["timings"]["ridgefill"]
        self.assertEqual(evaluation.validate_ridgefill_report(report, 700, "speed")["n_filled"], 100)
        for key, value in (("tokens_skipped", 0), ("n_exact_prefix", 1), ("mac_share_skipped", 0.9)):
            bad = dict(report, **{key: value})
            with self.assertRaises(evaluation.EvalError):
                evaluation.validate_ridgefill_report(bad, 700, "speed")

    def test_aggregate_reporting_weights_actual_row_counts(self):
        a = evaluation.response_ridgefill_report(dict(timings={}), 200, "off")
        b = evaluation.response_ridgefill_report(dict(timings={}), 100, "off")
        total = evaluation.reporting_summary([dict(reference=dict(ridgefill=a)), dict(reference=dict(ridgefill=b))], "reference")
        assert total is not None
        self.assertEqual(total["n_prompt"], 300)
        self.assertEqual(total["requests"], 2)
        self.assertEqual(total["tokens_skipped"], 0)
        self.assertIsNone(total["macs_exact"])
        first = dict(a, n_prefix=50, n_filled=50, n_exact_tail=150, tokens_skipped=0.25,
                     macs_exact=100.0, macs_ridgefill=80.0, mac_share_skipped=0.2)
        second = dict(b, n_prefix=100, n_filled=100, n_exact_tail=0, tokens_skipped=1.0,
                      macs_exact=200.0, macs_ridgefill=180.0, mac_share_skipped=0.1)
        weighted = evaluation.reporting_summary([dict(candidate=dict(ridgefill=first)), dict(candidate=dict(ridgefill=second))], "candidate")
        assert weighted is not None
        self.assertEqual(weighted["tokens_skipped"], 0.5)
        self.assertAlmostEqual(weighted["mac_share_skipped"], 40 / 300)
        with self.assertRaisesRegex(evaluation.EvalError, "missing RidgeFill"):
            evaluation.reporting_summary([dict(reference=dict(ridgefill=a)), dict(reference={})], "reference")

    def test_negative_off_uses_no_fill(self):
        result = evaluation.score_nll(FakeClient(), [i % 17 for i in range(700)], "off", 600, 1, 1)
        self.assertEqual(result["filled"], 0)
        self.assertEqual(result["prefix"], 0)

    def test_s1_reporting_sources_costs_and_legacy_aggregation(self):
        report: dict = dict(n_prompt=700, n_prefix=100, n_filled=97, n_exact_prefix=3, n_cached_exact_prefix=0, n_exact_tail=600,
                      n_exact_prefix_by_source={"class": 0, "digit": 0, "special": 1, "header": 0, "qattn": 0, "s1": 2},
                      tokens_skipped=97/700, macs_exact=100.0, macs_ridgefill=80.0, mac_share_skipped=0.2,
                      s1=dict(window=[650, 700], macs_fill=10, macs_probe=20, fill_recomputed=True),
                      vtail=dict(rows=32, macs_measure=5))
        self.assertEqual(evaluation.validate_ridgefill_report(report, 700, "quality+qattn"), report)
        legacy = dict(report, n_filled=100, n_exact_prefix=0, tokens_skipped=100/700,
                      n_exact_prefix_by_source=dict.fromkeys(("class", "digit", "special", "header", "qattn"), 0))
        del legacy["s1"], legacy["vtail"]
        total = evaluation.reporting_summary([dict(candidate=dict(ridgefill=report)), dict(candidate=dict(ridgefill=legacy))], "candidate")
        assert total is not None
        self.assertEqual(total["n_exact_prefix_by_source"]["s1"], 2)
        self.assertEqual(sum(total["n_exact_prefix_by_source"].values()), 3)
        for key, values in (("s1", {"window": [99, 149]}), ("s1", {"macs_fill": float("nan")}),
                            ("s1", {"fill_recomputed": False}), ("vtail", {"rows": 601}), ("vtail", {"macs_measure": -1})):
            bad = copy.deepcopy(report)
            bad[key].update(values)
            with self.assertRaises(evaluation.EvalError):
                evaluation.validate_ridgefill_report(bad, 700, "quality+qattn")
        bad = copy.deepcopy(report)
        del bad["s1"]
        with self.assertRaises(evaluation.EvalError):
            evaluation.validate_ridgefill_report(bad, 700, "quality+qattn")

    def test_scored_rows_never_filled(self):
        with self.assertRaisesRegex(evaluation.EvalError, "tail-512"):
            evaluation.score_nll(FakeClient(), list(range(700)), "speed", 200, 1, 1)

    def test_needle_answer_suffix_removed(self):
        client = FakeClient()
        item = dict(ids=[10, 11, 12, 90, 91], answer_len=2)
        evaluation.needle_prompt(client, item)
        self.assertEqual(client.calls[0], ("/detokenize", dict(tokens=[10, 11, 12])))
        self.assertEqual(client.calls[1][1]["chat_template_kwargs"], dict(enable_thinking=False))

    def test_wrong_token_and_post_sampling_refused(self):
        with self.assertRaisesRegex(evaluation.EvalError, "target token"):
            evaluation.target_logprob(dict(completion_probabilities=[dict(id=7, logprob=-1)]), 8)
        with self.assertRaisesRegex(evaluation.EvalError, "pre-sampling"):
            evaluation.target_logprob(dict(completion_probabilities=[dict(id=8, prob=1)]), 8)

    def test_needle_numeric_boundaries(self):
        self.assertTrue(evaluation.needle_correct("The answer is 1234567.", "1234567"))
        self.assertFalse(evaluation.needle_correct("12345678", "1234567"))
        self.assertFalse(evaluation.needle_correct("91234567", "1234567"))
        self.assertTrue(evaluation.needle_correct("0000001", "0000001"))
        for value in ("123456", "12345678", "abcdefg", "123456\u0667"):
            with self.assertRaisesRegex(evaluation.EvalError, "seven ASCII digits"):
                evaluation.needle_correct(value, value)

    def test_needle_requires_qwen_thinking_off(self):
        class ThinkingClient(FakeClient):
            def request(self, path: str, body: dict | None = None) -> dict:
                result = super().request(path, body)
                if path == "/apply-template":
                    result["prompt"] = "<|im_start|>assistant\n<think>\n"
                return result
        with self.assertRaisesRegex(evaluation.EvalError, "thinking-off"):
            evaluation.needle_prompt(ThinkingClient(), dict(ids=[10, 11, 90], answer_len=1))

    def test_inactive_fill_refused(self):
        with self.assertRaisesRegex(evaluation.EvalError, "did not fill"):
            evaluation.check_fill(dict(timings={}), expected=True, mode="speed")
        with self.assertRaisesRegex(evaluation.EvalError, "mode off"):
            evaluation.check_fill(dict(timings=dict(prompt_filled_n=1)), expected=False, mode="off")


if __name__ == "__main__":
    unittest.main()
