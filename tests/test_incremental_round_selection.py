import argparse
import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


def _stub_module(name: str, **attrs: object) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules.setdefault(name, module)


def _install_import_stubs() -> None:
    def _unexpected(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("unexpected call into test stub")

    if "cv2" not in sys.modules:
        _stub_module(
            "cv2",
            COLOR_RGB2HSV=0,
            COLOR_RGB2GRAY=1,
            MORPH_ELLIPSE=0,
            MORPH_OPEN=1,
            MORPH_CLOSE=2,
            THRESH_BINARY=0,
            THRESH_OTSU=0,
            cvtColor=_unexpected,
            threshold=_unexpected,
            bitwise_and=_unexpected,
            getStructuringElement=_unexpected,
            morphologyEx=_unexpected,
            calcHist=_unexpected,
            Canny=_unexpected,
        )

    if "skimage" not in sys.modules:
        _stub_module("skimage")
    if "skimage.feature" not in sys.modules:
        _stub_module(
            "skimage.feature",
            graycomatrix=_unexpected,
            graycoprops=_unexpected,
            local_binary_pattern=_unexpected,
        )
    if "skimage.measure" not in sys.modules:
        _stub_module("skimage.measure", shannon_entropy=_unexpected)

    if "sklearn" not in sys.modules:
        _stub_module("sklearn")
    if "sklearn.decomposition" not in sys.modules:
        class _PCA:
            def __init__(self, n_components: int, random_state: int | None = None, svd_solver: str = "full"):
                self.n_components = n_components

            def fit_transform(self, X):  # type: ignore[no-untyped-def]
                arr = np.asarray(X, dtype=float)
                return arr[:, : self.n_components]

        _stub_module("sklearn.decomposition", PCA=_PCA)
    if "sklearn.metrics" not in sys.modules:
        _stub_module("sklearn.metrics")
    if "sklearn.metrics.pairwise" not in sys.modules:
        _stub_module("sklearn.metrics.pairwise", cosine_distances=_unexpected)
    if "sklearn.preprocessing" not in sys.modules:
        class _StandardScaler:
            def fit_transform(self, X):  # type: ignore[no-untyped-def]
                return np.asarray(X, dtype=float)

        _stub_module("sklearn.preprocessing", StandardScaler=_StandardScaler)


_install_import_stubs()

import select_diverse_wsi


class IncrementalRoundSelectionTest(unittest.TestCase):
    def test_parse_args_rejects_removed_version_flag(self) -> None:
        with mock.patch.object(sys, "argv", ["select_diverse_wsi.py", "--input_dir", "/tmp", "--version"]):
            with self.assertRaises(SystemExit) as exc_info:
                select_diverse_wsi.parse_args()

        self.assertEqual(exc_info.exception.code, 2)

    def test_run_appends_new_round_with_relative_paths_and_migrates_legacy_history(self) -> None:
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
            self.assertEqual(
                result["path"].tolist(),
                ["tissueA/a.svs", "tissueA/b.svs", "tissueA/c.svs"],
            )
            self.assertEqual(result["global_rank"].tolist()[1:], [1, 2])

            output_text = stdout.getvalue()
            self.assertIn("经过 2 轮挑选，共选过 3 张 WSI，本轮还剩 2 张 WSI，已选择 2 张", output_text)
            self.assertIn("因剩余数量'2'小于每轮挑选设定数量'5'，已全部挑选", output_text)

    def test_run_writes_failed_csv_with_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            args = argparse.Namespace(
                input_dir=str(input_dir),
                extensions="svs",
                thumb_side=64,
                top_frac=0.5,
                min_per_tissue=1,
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

            bad_path = str(input_dir / "tissueA" / "bad.svs")

            with mock.patch.object(select_diverse_wsi, "discover_wsi_paths", return_value=[bad_path]):
                code = select_diverse_wsi.run(args)

            self.assertEqual(code, 3)
            failed = pd.read_csv(output_dir / "failed_wsi.csv")
            self.assertEqual(failed["path"].tolist(), ["tissueA/bad.svs"])
            self.assertEqual(failed["tissue_type"].tolist(), ["tissueA"])

    def test_run_raises_when_legacy_history_path_is_outside_input_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            input_dir.mkdir()
            output_dir = root / "output"
            output_dir.mkdir()

            selected_csv = output_dir / "selected_wsi.csv"
            outside_root = root / "outside"
            outside_root.mkdir()
            outside_path = outside_root / "foreign.svs"

            pd.DataFrame(
                [
                    {
                        "round": 1,
                        "tissue_type": "tissueA",
                        "tissue_rank": 1,
                        "global_rank": 1,
                        "path": str(outside_path),
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
                min_per_tissue=1,
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

            with self.assertRaisesRegex(ValueError, "outside the current input_dir"):
                select_diverse_wsi.run(args)


if __name__ == "__main__":
    unittest.main()
