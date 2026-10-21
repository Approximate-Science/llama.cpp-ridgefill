"""CPU parser/refusal controls for GPU evidence; synthetic records do not execute a GPU gate."""

import copy
import unittest

from check_gpu_smoke import select_devices, validate_bench


class SmokeEvidence(unittest.TestCase):
    def setUp(self):
        self.devices = ["Vulkan" + str(index) for index in range(2)]
        self.inventory = "Available devices:\n" + "".join(
            f"  {device}: AMD Radeon AI PRO 9700 (32768 MiB, 30000 MiB free)\n" for device in self.devices)
        self.row = dict(n_prompt=8192, n_gen=0, n_depth=0, n_gpu_layers=99, split_mode="layer", type_k="f16",
                        type_v="f16", flash_attn=0, build_commit="123abcdef", devices="/".join(self.devices),
                        backends="Vulkan", avg_ts=100.0, avg_ns=81920000000, samples_ns=[81920000000], samples_ts=[100.0])
        self.marker = "RidgeFill: mode speed completed 4096 prefix rows, 4096 exact tail rows, 1 measured repetitions"
        self.buffers = [f"load_tensors: {device} model buffer size = 8000.00 MiB" for device in self.devices]

    def validate(self, row=None, log=None):
        return validate_bench([self.row if row is None else row], self.marker + "\n" + "\n".join(self.buffers)
                              if log is None else log, self.devices, "123abcdef", "vulkan")

    def test_evidence_positive_and_missing_gpu_buffer(self):
        self.assertEqual(set(self.validate()["model_buffer_mib"]), set(self.devices))
        with self.assertRaisesRegex(ValueError, "place model tensors"):
            self.validate(log=self.marker + "\n" + self.buffers[0])

    def test_cpu_fallback_and_inactive_prefix(self):
        row = copy.deepcopy(self.row)
        row["backends"] = "CPU"
        with self.assertRaisesRegex(ValueError, "CPU fallback"):
            self.validate(row=row)
        with self.assertRaisesRegex(ValueError, "active completed speed prefix"):
            self.validate(log="\n".join(self.buffers))

    def test_wrong_revision_devices_mode_and_invalid_samples(self):
        for key, value in (("build_commit", "other"), ("devices", self.devices[0]), ("split_mode", "row"),
                           ("n_gpu_layers", 0), ("n_depth", 512), ("avg_ts", float("nan")), ("samples_ns", [0])):
            with self.subTest(key=key):
                row = copy.deepcopy(self.row)
                row[key] = value
                with self.assertRaises(ValueError):
                    self.validate(row=row)

    def test_two_cards_requires_both_r9700s(self):
        self.assertEqual(select_devices(self.inventory, "vulkan", True), self.devices)
        one = "\n".join(self.inventory.splitlines()[:2])
        for inventory in (one, "Available devices:\n  (none)\n", self.inventory.replace("9700", "other")):
            with self.subTest(inventory=inventory):
                with self.assertRaisesRegex(ValueError, "exactly 2"):
                    select_devices(inventory, "vulkan", True)

    def test_single_card_bridge_and_headroom(self):
        one = "\n".join(self.inventory.splitlines()[:2])
        self.assertEqual(select_devices(one, "vulkan", False), self.devices[:1])
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            select_devices(self.inventory, "vulkan", False)
        with self.assertRaisesRegex(ValueError, "3072 MiB"):
            select_devices(one.replace("30000 MiB free", "1024 MiB free"), "vulkan", False)
        hip = one.replace(self.devices[0], "ROCm" + str(0))
        self.assertEqual(select_devices(hip, "hip", False), ["ROCm" + str(0)])


if __name__ == "__main__":
    unittest.main()
