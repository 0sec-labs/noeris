from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class CrossVendorMeasuredPackFromPredictionTests(unittest.TestCase):
    def test_generates_measured_template(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pred = root / "pred.json"
            out = root / "measured.json"
            pred.write_text(
                json.dumps(
                    {
                        "predictions": {
                            "attention": {
                                "bucket_a": {
                                    "target_predicted_top": [
                                        {"config_id": "c1"},
                                        {"config_id": "c2"},
                                    ]
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            cmd = [
                "python3",
                "scripts/cross_vendor_measured_pack_from_prediction.py",
                "--prediction-json",
                str(pred),
                "--output-json",
                str(out),
                "--top-k",
                "2",
            ]
            subprocess.run(cmd, check=True)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["artifact_type"], "cross_vendor_measured_template")
            self.assertEqual(data["measurement_status"], "placeholder_not_measured")
            self.assertFalse(data["is_measured_evidence"])
            rows = data["measured"]["attention"]["bucket_a"]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["config_id"], "c1")
            self.assertEqual(rows[0]["metric"], 0.0)
            self.assertEqual(rows[0]["latency_ms"], 0.0)
            self.assertIn("placeholder", rows[0]["notes"])


if __name__ == "__main__":
    unittest.main()
