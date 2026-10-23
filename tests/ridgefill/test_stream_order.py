import copy
import math
import unittest

from check_stream_order import compare, float_bytes


class StreamOrderComparison(unittest.TestCase):
    def setUp(self):
        self.row = dict(logprobs=[-1.0, -2.0], logsumexp=[3.0, 4.0], s1_mass=[0.25, 0.5],
                        exact_rows=[0], target_positions=[2, 3], n=4, scored=2, prefills=1)

    def test_exact(self):
        self.assertTrue(all(compare(self.row, copy.deepcopy(self.row)).values()))

    def test_one_ulp_and_signed_zero(self):
        self.assertNotEqual(float_bytes([1.0]), float_bytes([math.nextafter(1.0, 2.0)]))
        self.assertNotEqual(float_bytes([0.0]), float_bytes([-0.0]))

    def test_changed_score_and_selection(self):
        changed = copy.deepcopy(self.row)
        changed["logprobs"][0] = math.nextafter(-1.0, 0.0)
        changed["exact_rows"] = [1]
        result = compare(self.row, changed)
        self.assertFalse(result["logprobs"])
        self.assertFalse(result["exact_rows"])

    def test_missing_and_nonfinite(self):
        for values in ([], [math.nan], [math.inf]):
            with self.assertRaises(ValueError):
                float_bytes(values)


if __name__ == "__main__":
    unittest.main()
