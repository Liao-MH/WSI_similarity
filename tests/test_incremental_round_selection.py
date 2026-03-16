import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import select_diverse_wsi


class IncrementalRoundSelectionTest(unittest.TestCase):
    def test_run_appends_new_round_and_skips_previously_selected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            input_dir.mkdir()
            output_dir = root / "output"
            output_dir.mkdir()

            selected_csv = output_dir / "selected_wsi.csv"
            pd.DataFrame(
                [
                    {
                        "round": 1,
                        "tissue_type": "tissueA",
                        "tissue_rank": 1,
                        "global_rank": 1,
                        "path": str(input_dir / "tissueA" / "a.svs"),
                        "selected_by": "kcenter",
                        "mean_cosine_distance": 0.4,
                        "tissue_ratio": 0.7,
                        "mask_fallback": 0,
                        "group_total": 1,
                        "group_selected": 1,
                    }
                ]
            ).to_csv(selected_csv, index=False)

            args = argparse.Namespace(
                input_dir=str(input_dir),
                extensions="svs",
                thumb_side=64,
                top_frac=0.5,
                min_per_tissue=5,
                pca_dim=2,
                hsv_bins=4,
                glcm_levels=8,
                lbp_p=8,
                lbp_r=1.0,
                seed=42,
                output_dir=str(output_dir),
                out_csv="selected_wsi.csv",
                out_failed_csv="failed_wsi.csv",
                cache_dir="thumb_cache",
            )

            all_paths = [
                str(input_dir / "tissueA" / "a.svs"),
                str(input_dir / "tissueA" / "b.svs"),
                str(input_dir / "tissueA" / "c.svs"),
            ]
            feature_rows = iter([[1.0, 0.0], [0.0, 1.0]])

            with (
                mock.patch.object(select_diverse_wsi, "discover_wsi_paths", return_value=all_paths),
                mock.patch.object(select_diverse_wsi, "load_thumbnail", return_value=None),
                mock.patch.object(
                    select_diverse_wsi,
                    "build_tissue_mask",
                    return_value=(None, 0.8, False),
                ),
                mock.patch.object(
                    select_diverse_wsi,
                    "extract_features",
                    side_effect=lambda *args, **kwargs: next(feature_rows),
                ),
                mock.patch.object(
                    select_diverse_wsi,
                    "kcenter_fps_select",
                    return_value=([0, 1], [0.3, 0.2]),
                ),
            ):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = select_diverse_wsi.run(args)

            self.assertEqual(code, 0)

            result = pd.read_csv(selected_csv)
            self.assertEqual(len(result), 3)
            self.assertEqual(result["round"].tolist(), [1, 2, 2])
            self.assertEqual(result["path"].tolist()[1:], all_paths[1:])
            self.assertEqual(result["global_rank"].tolist()[1:], [1, 2])

            output_text = stdout.getvalue()
            self.assertIn("经过 2 轮挑选，共选过 3 张 WSI，本轮还剩 2 张 WSI，已选择 2 张", output_text)
            self.assertIn("因剩余数量'2'小于每轮挑选设定数量'5'，已全部挑选", output_text)


if __name__ == "__main__":
    unittest.main()
