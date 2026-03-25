#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

SCRIPT_VERSION = "v2.0.1"

try:
    import openslide  # type: ignore
except Exception:
    openslide = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select diverse WSI samples via k-center/FPS on handcrafted features."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="WSI root directory.")
    parser.add_argument(
        "--extensions",
        type=str,
        default="svs,tif,tiff",
        help="Comma-separated extensions to auto-discover recursively.",
    )
    parser.add_argument("--thumb_side", type=int, default=512, help="Max side for thumbnails.")
    parser.add_argument("--top_frac", type=float, default=0.10, help="Fraction to select.")
    parser.add_argument(
        "--min_per_tissue",
        type=int,
        default=5,
        help="Minimum selected count per tissue type (capped by group size).",
    )
    parser.add_argument("--pca_dim", type=int, default=32, help="PCA embedding dimension.")
    parser.add_argument("--hsv_bins", type=int, default=16, help="HSV histogram bins per channel.")
    parser.add_argument("--glcm_levels", type=int, default=32, help="GLCM levels.")
    parser.add_argument("--lbp_p", type=int, default=8, help="LBP P parameter.")
    parser.add_argument("--lbp_r", type=float, default=1.0, help="LBP R parameter.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="output", help="Unified output directory.")
    parser.add_argument("--out_csv", type=str, default="selected_wsi.csv", help="Output selected CSV filename.")
    parser.add_argument("--out_failed_csv", type=str, default="failed_wsi.csv", help="Failed CSV filename.")
    parser.add_argument("--cache_dir", type=str, default="thumb_cache", help="Thumbnail cache subdirectory.")
    return parser.parse_args()


def discover_wsi_paths(input_dir: str, extensions: str) -> List[str]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    exts = {("." + e.strip().lower().lstrip(".")) for e in extensions.split(",") if e.strip()}
    if not exts:
        raise ValueError("No valid extensions provided.")
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            suffix = Path(name).suffix.lower()
            if suffix in exts:
                out.append(str(Path(dirpath) / name))
    return sorted(out)


def infer_tissue_type(path: str, input_dir: str) -> str:
    rel = Path(path).resolve().relative_to(Path(input_dir).resolve())
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return "__root__"


def ensure_output_paths(args: argparse.Namespace) -> argparse.Namespace:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args.out_csv = str(output_dir / Path(args.out_csv).name)
    if args.out_failed_csv:
        args.out_failed_csv = str(output_dir / Path(args.out_failed_csv).name)
    if args.cache_dir:
        args.cache_dir = str(output_dir / Path(args.cache_dir).name)
    return args


def resolve_input_dir(input_dir: str) -> Path:
    root = Path(input_dir).expanduser().resolve(strict=False)
    if not root.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    return root


def canonical_relative_wsi_path(path: str, input_dir: Path) -> str:
    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        abs_path = raw_path.resolve(strict=False)
    else:
        abs_path = (input_dir / raw_path).resolve(strict=False)

    # Canonicalize every path against input_dir so CSV history stays stable across devices.
    try:
        rel_path = abs_path.relative_to(input_dir)
    except ValueError as exc:
        raise ValueError(f"WSI path is outside the current input_dir: {path}") from exc
    return rel_path.as_posix()


def load_selection_history(out_csv: str, input_dir: str) -> Tuple[pd.DataFrame, Set[str], int]:
    csv_path = Path(out_csv)
    if not csv_path.exists():
        return pd.DataFrame(), set(), 1

    history_df = pd.read_csv(csv_path)
    if history_df.empty:
        return history_df, set(), 1

    if "path" not in history_df.columns:
        raise ValueError(f"Selection history is missing required 'path' column: {out_csv}")

    rewrite_history = False
    if "round" not in history_df.columns:
        history_df.insert(0, "round", 1)
        rewrite_history = True

    history_df["round"] = history_df["round"].fillna(1).astype(int)
    input_root = resolve_input_dir(input_dir)
    normalized_paths: List[str] = []
    for raw_path in history_df["path"].astype(str).tolist():
        normalized_path = canonical_relative_wsi_path(raw_path, input_root)
        normalized_paths.append(normalized_path)
        if raw_path != normalized_path:
            rewrite_history = True

    history_df["path"] = normalized_paths
    if rewrite_history:
        history_df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    selected_paths = set(normalized_paths)
    next_round = int(history_df["round"].max()) + 1
    return history_df, selected_paths, next_round


def _thumb_cache_path(cache_dir: str, src_path: str, thumb_side: int) -> Path:
    sanitized = src_path.replace("\\", "_").replace("/", "_").replace(":", "_")
    return Path(cache_dir) / f"{sanitized}.s{thumb_side}.jpg"


def _load_thumbnail_via_openslide(path: str, thumb_side: int) -> Optional[np.ndarray]:
    if openslide is None:
        return None
    try:
        slide = openslide.OpenSlide(path)
        w, h = slide.dimensions
        if w <= 0 or h <= 0:
            return None
        scale = thumb_side / float(max(w, h))
        tw = max(1, int(round(w * scale)))
        th = max(1, int(round(h * scale)))
        thumb = slide.get_thumbnail((tw, th)).convert("RGB")
        arr = np.array(thumb, dtype=np.uint8)
        slide.close()
        return arr
    except Exception:
        return None


def _load_thumbnail_via_pil(path: str, thumb_side: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((thumb_side, thumb_side), Image.Resampling.BILINEAR)
        return np.array(im, dtype=np.uint8)


def load_thumbnail(path: str, thumb_side: int, cache_dir: str = "") -> np.ndarray:
    if cache_dir:
        cache_path = _thumb_cache_path(cache_dir, path, thumb_side)
        if cache_path.exists():
            with Image.open(cache_path) as im:
                return np.array(im.convert("RGB"), dtype=np.uint8)

    thumb = _load_thumbnail_via_openslide(path, thumb_side)
    if thumb is None:
        thumb = _load_thumbnail_via_pil(path, thumb_side)

    if cache_dir:
        cache_path = _thumb_cache_path(cache_dir, path, thumb_side)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(thumb).save(cache_path, quality=90)
    return thumb


def build_tissue_mask(rgb: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    _, s_bin = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v_bin = (v < 245).astype(np.uint8) * 255
    mask = cv2.bitwise_and(s_bin, v_bin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_bool = mask > 0
    tissue_ratio = float(mask_bool.mean())

    used_fallback = False
    if tissue_ratio < 0.01:
        mask_bool = np.ones(mask_bool.shape, dtype=bool)
        tissue_ratio = 0.0
        used_fallback = True
    return mask_bool, tissue_ratio, used_fallback


def _safe_stats(x: np.ndarray) -> Tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    return float(np.mean(x)), float(np.std(x))


def extract_features(
    rgb: np.ndarray,
    mask: np.ndarray,
    tissue_ratio: float,
    hsv_bins: int,
    glcm_levels: int,
    lbp_p: int,
    lbp_r: float,
) -> np.ndarray:
    feats: List[float] = []
    feats.append(float(tissue_ratio))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    rgb_f = rgb.astype(np.float32)
    hsv_f = hsv.astype(np.float32)

    for c in range(3):
        m, s = _safe_stats(rgb_f[:, :, c][mask])
        feats.extend([m, s])
    for c in range(3):
        m, s = _safe_stats(hsv_f[:, :, c][mask])
        feats.extend([m, s])

    for c in range(3):
        vals = hsv[:, :, c][mask]
        if vals.size == 0:
            hist = np.zeros((hsv_bins,), dtype=np.float32)
        else:
            max_val = 180 if c == 0 else 256
            hist = cv2.calcHist([vals.reshape(-1, 1)], [0], None, [hsv_bins], [0, max_val]).flatten()
            hist = hist.astype(np.float32)
            if hist.sum() > 0:
                hist /= hist.sum()
        feats.extend(hist.tolist())

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=lbp_p, R=lbp_r, method="uniform")
    n_bins = lbp_p + 2
    lbp_vals = lbp[mask]
    if lbp_vals.size == 0:
        lbp_hist = np.zeros((n_bins,), dtype=np.float32)
    else:
        lbp_hist, _ = np.histogram(lbp_vals, bins=np.arange(0, n_bins + 1), density=True)
        lbp_hist = lbp_hist.astype(np.float32)
    feats.extend(lbp_hist.tolist())

    q = (gray.astype(np.float32) / 256.0 * glcm_levels).astype(np.uint8)
    q = np.clip(q, 0, glcm_levels - 1)
    glcm = graycomatrix(
        q,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=glcm_levels,
        symmetric=True,
        normed=True,
    )
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        vals = graycoprops(glcm, prop).reshape(-1)
        feats.extend([float(np.mean(vals)), float(np.std(vals))])

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0)[mask].mean()) if np.any(mask) else 0.0
    feats.append(edge_density)
    feats.append(float(shannon_entropy(gray)))
    return np.asarray(feats, dtype=np.float32)


def kcenter_fps_select(X: np.ndarray, k: int) -> Tuple[List[int], np.ndarray]:
    if X.shape[0] == 0:
        return [], np.array([], dtype=np.float32)
    dmat = cosine_distances(X, X)
    mean_dist = dmat.mean(axis=1)
    first = int(np.argmax(mean_dist))

    selected = [first]
    min_dist = dmat[first].copy()
    min_dist[first] = -np.inf

    while len(selected) < k:
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, dmat[nxt])
        min_dist[selected] = -np.inf
    return selected, mean_dist


def run(args: argparse.Namespace) -> int:
    np.random.seed(args.seed)
    t0 = time.time()
    args = ensure_output_paths(args)
    input_root = resolve_input_dir(args.input_dir)
    history_df, previously_selected_paths, current_round = load_selection_history(args.out_csv, args.input_dir)

    discovered_paths = discover_wsi_paths(args.input_dir, args.extensions)
    if not discovered_paths:
        print("No input files found.", file=sys.stderr)
        return 2

    path_records = [
        (canonical_relative_wsi_path(path, input_root), path) for path in discovered_paths
    ]
    path_records = sorted(path_records, key=lambda item: item[0])
    remaining_records = [
        (relative_path, absolute_path)
        for relative_path, absolute_path in path_records
        if relative_path not in previously_selected_paths
    ]

    tissue_to_paths: Dict[str, List[Tuple[str, str]]] = {}
    for relative_path, absolute_path in remaining_records:
        tissue = infer_tissue_type(absolute_path, args.input_dir)
        tissue_to_paths.setdefault(tissue, []).append((relative_path, absolute_path))

    round_remaining_total = len(remaining_records)
    planned_round_total = sum(
        max(args.min_per_tissue, int(math.ceil(args.top_frac * len(group_paths))))
        for group_paths in tissue_to_paths.values()
    )

    if round_remaining_total == 0:
        total_selected = len(history_df.index)
        print(
            f"经过 {current_round - 1} 轮挑选，共选过 {total_selected} 张 WSI，本轮还剩 0 张 WSI，已选择 0 张"
        )
        print(
            f"[SUMMARY] total={len(path_records)} ok=0 failed=0 selected=0 warnings=0 "
            f"total_time={time.time() - t0:.3f}s avg_time=0.000s out_dir={args.output_dir}"
        )
        return 0

    features: List[np.ndarray] = []
    ok_rows: List[Dict[str, object]] = []
    failed_rows: List[Dict[str, str]] = []
    warnings = 0

    tissue_names = sorted(tissue_to_paths.keys())
    for tissue_idx, tissue in enumerate(tissue_names, start=1):
        tissue_paths = sorted(tissue_to_paths[tissue], key=lambda item: item[0])
        tissue_start = time.time()
        tissue_ok = 0
        tissue_fail = 0
        print(f"[TISSUE] {tissue_idx}/{len(tissue_names)} tissue={tissue} total={len(tissue_paths)}")
        progress_bar = tqdm(
            tissue_paths,
            desc=f"{tissue}",
            unit="wsi",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]",
        )
        for relative_path, absolute_path in progress_bar:
            try:
                thumb = load_thumbnail(absolute_path, args.thumb_side, args.cache_dir)
                mask, tissue_ratio, used_fallback = build_tissue_mask(thumb)
                feat = extract_features(
                    thumb,
                    mask,
                    tissue_ratio=tissue_ratio if not used_fallback else 0.0,
                    hsv_bins=args.hsv_bins,
                    glcm_levels=args.glcm_levels,
                    lbp_p=args.lbp_p,
                    lbp_r=args.lbp_r,
                )
                features.append(feat)
                ok_rows.append(
                    {
                        "path": relative_path,
                        "tissue_type": tissue,
                        "tissue_ratio": tissue_ratio,
                        "mask_fallback": int(used_fallback),
                    }
                )
                tissue_ok += 1
                if used_fallback:
                    warnings += 1
            except Exception as e:
                msg = str(e).replace("\n", " ")
                failed_rows.append({"path": relative_path, "tissue_type": tissue, "error": msg})
                tissue_fail += 1
        tissue_elapsed = time.time() - tissue_start
        print(
            f"[TISSUE_DONE] tissue={tissue} ok={tissue_ok} failed={tissue_fail} "
            f"time={tissue_elapsed:.2f}s"
        )

    n_ok = len(ok_rows)
    if n_ok == 0:
        print("All files failed during feature extraction.", file=sys.stderr)
        if args.out_failed_csv:
            pd.DataFrame(failed_rows).to_csv(args.out_failed_csv, index=False)
        return 3

    F = np.vstack(features).astype(np.float32)
    tissue_to_indices: Dict[str, List[int]] = {}
    for idx, row in enumerate(ok_rows):
        tissue = str(row["tissue_type"])
        tissue_to_indices.setdefault(tissue, []).append(idx)

    out_rows: List[Dict[str, object]] = []
    for tissue in sorted(tissue_to_indices.keys()):
        group_indices = tissue_to_indices[tissue]
        group_n = len(group_indices)
        target_k = max(args.min_per_tissue, int(math.ceil(args.top_frac * group_n)))
        k = min(group_n, target_k)

        group_F = F[group_indices]
        scaler = StandardScaler()
        group_Fz = scaler.fit_transform(group_F)
        pca_dim = max(1, min(args.pca_dim, group_Fz.shape[0], group_Fz.shape[1]))
        group_X = PCA(n_components=pca_dim, svd_solver="full", random_state=args.seed).fit_transform(group_Fz)
        local_selected, group_mean_dist = kcenter_fps_select(group_X, k)

        if len(local_selected) < k:
            print(f"Selection failed for tissue={tissue}.", file=sys.stderr)
            return 4

        print(f"[GROUP] tissue={tissue} total={group_n} selected={k}")
        for tissue_rank, local_idx in enumerate(local_selected, start=1):
            global_idx = group_indices[local_idx]
            out_rows.append(
                {
                    "round": current_round,
                    "tissue_type": tissue,
                    "tissue_rank": tissue_rank,
                    "path": ok_rows[global_idx]["path"],
                    "selected_by": "kcenter",
                    "mean_cosine_distance": float(group_mean_dist[local_idx]),
                    "tissue_ratio": float(ok_rows[global_idx]["tissue_ratio"]),
                    "mask_fallback": int(ok_rows[global_idx]["mask_fallback"]),
                    "group_total": group_n,
                    "group_selected": k,
                }
            )

    out_rows = sorted(out_rows, key=lambda x: (str(x["tissue_type"]), int(x["tissue_rank"])))
    for i, row in enumerate(out_rows, start=1):
        row["global_rank"] = i
    selected_total = len(out_rows)
    current_df = pd.DataFrame(out_rows)
    combined_df = pd.concat([history_df, current_df], ignore_index=True, sort=False)
    pd.DataFrame(combined_df).to_csv(args.out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    if args.out_failed_csv:
        pd.DataFrame(failed_rows).to_csv(args.out_failed_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    elapsed = time.time() - t0
    avg = elapsed / max(1, len(path_records))
    cumulative_selected_total = len(combined_df.index)
    if selected_total > 0:
        summary = (
            f"经过 {current_round} 轮挑选，共选过 {cumulative_selected_total} 张 WSI，"
            f"本轮还剩 {round_remaining_total} 张 WSI，已选择 {selected_total} 张"
        )
        if round_remaining_total < planned_round_total:
            summary += (
                f"，因剩余数量'{round_remaining_total}'小于每轮挑选设定数量"
                f"'{planned_round_total}'，已全部挑选"
            )
        print(summary)
    print(
        f"[SUMMARY] total={len(path_records)} ok={n_ok} failed={len(failed_rows)} "
        f"selected={selected_total} warnings={warnings} "
        f"total_time={elapsed:.3f}s avg_time={avg:.3f}s out_dir={args.output_dir}"
    )
    return 0


def main() -> None:
    args = parse_args()
    code = run(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
