"""CPU parser/refusal controls for GPU evidence; synthetic records do not execute a GPU gate."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from check_gpu_smoke import select_devices, validate_bench
from run_g5 import check_sdk_version, grep


class SmokeEvidence(unittest.TestCase):
    def setUp(self):
        self.devices = ["Vulkan" + str(index) for index in range(2)]
        self.inventory = "Available devices:\n" + "".join(
            f"  {device}: Example GPU {index} (32768 MiB, 30000 MiB free)\n" for index, device in enumerate(self.devices))
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

    def test_two_cards_accept_any_models_and_optional_name_filter(self):
        self.assertEqual(select_devices(self.inventory, "vulkan", True), self.devices)
        one = "\n".join(self.inventory.splitlines()[:2])
        for inventory in (one, "Available devices:\n  (none)\n"):
            with self.subTest(inventory=inventory):
                with self.assertRaisesRegex(ValueError, "exactly 2"):
                    select_devices(inventory, "vulkan", True)
        self.assertEqual(select_devices(self.inventory, "vulkan", True, "Example"), self.devices)
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            select_devices(self.inventory, "vulkan", True, "different model")
        self.assertEqual(select_devices(self.inventory, "vulkan", False, "Example GPU " + str(1)), self.devices[1:])

    def test_single_card_bridge_and_headroom(self):
        one = "\n".join(self.inventory.splitlines()[:2])
        self.assertEqual(select_devices(one, "vulkan", False), self.devices[:1])
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            select_devices(self.inventory, "vulkan", False)
        with self.assertRaisesRegex(ValueError, "3072 MiB"):
            select_devices(one.replace("30000 MiB free", "1024 MiB free"), "vulkan", False)
        hip = one.replace(self.devices[0], "ROCm" + str(0))
        self.assertEqual(select_devices(hip, "hip", False), ["ROCm" + str(0)])

    def test_sdk_backend_floor_and_optional_higher_minimum(self):
        cmake = "if (${hip_VERSION} VERSION_LESS 6.1)"
        def version(*parts):
            return ".".join(map(str, parts))
        check_sdk_version(version(6, 1, 0), cmake)
        check_sdk_version(version(9, 2, 0), cmake)
        with self.assertRaisesRegex(ValueError, "below"):
            check_sdk_version(version(5, 9, 0), cmake)
        with self.assertRaisesRegex(ValueError, "below"):
            check_sdk_version(version(6, 2, 0), cmake, version(8, 0))
        with self.assertRaisesRegex(ValueError, "below"):
            check_sdk_version(version(5, 9, 0), cmake, version(4, 0))
        with self.assertRaisesRegex(ValueError, "minimum HIP version"):
            check_sdk_version(version(9, 2, 0), "unsupported CMake layout")
        with self.assertRaisesRegex(ValueError, "must start"):
            check_sdk_version("unknown", cmake)

    def audit_controls(self):
        values = ["R9" + str(123), "Radeon AI PRO " + str(9000), "gfx" + str(1200), "gfx" + str(1100),
                  "ROCm " + ".".join(map(str, (8, 0, 0))), ".".join(map(str, (8, 0, 0))),
                  ":".join(("abcd", "12", "34.5")), ":".join(("56", "78.0"))]
        work = Path(__file__).resolve().parents[2] / "out"
        work.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as directory:
            path = Path(directory)
            sample = path / "nonportable.cpp"
            sample.write_text("\n".join(values) + "\n")
            found = grep([sample], path / "positive.log")
            self.assertEqual(found.returncode, 0)
            self.assertEqual(len(found.stdout.splitlines()), len(values))
            sample.write_text('const char * loopback = "127.0.0.1";\nint count(int argc) { return argc; }\n')
            found = grep([sample], path / "negative.log")
            self.assertEqual(found.returncode, 1)
            self.assertEqual(found.stdout, "")

    def test_extended_audit_positive_and_neutral_controls(self):
        self.audit_controls()

    def test_extended_audit_without_ripgrep(self):
        with patch("run_g5.subprocess.run", side_effect=FileNotFoundError("rg is absent")):
            self.audit_controls()


if __name__ == "__main__":
    unittest.main()
