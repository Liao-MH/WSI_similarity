# WSI Diversity Selection (k-center/FPS)

基于缩略图和手工特征（颜色 + 纹理 + 结构）对 WSI 做多样性优先筛选，输出覆盖面最大的 Top-K（默认 10%），用于优先标注。

## 1. 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`svs` 读取建议安装 OpenSlide 系统库（`openslide-python` 仅为 Python 绑定）。若环境无 OpenSlide，脚本会自动尝试用 PIL fallback（适用于多数 tif）。

## 2. 用法

```bash
python3 select_diverse_wsi.py \
  --input_dir /path/to/wsi_root \
  --extensions "svs,tif,tiff" \
  --thumb_side 512 \
  --top_frac 0.10 \
  --min_per_tissue 5 \
  --pca_dim 32 \
  --out_csv selected_wsi.csv \
  --out_failed_csv failed_wsi.csv \
  --cache_dir thumb_cache
```

脚本会递归扫描 `--input_dir` 下所有匹配后缀文件，并按一级子目录作为组织类型分组（例如 `Breast cancer N=137`）。

## 3. 关键参数

- `--input_dir`: WSI 目录
- `--extensions`: 自动识别后缀（逗号分隔），默认 `svs,tif,tiff`
- `--thumb_side`: 缩略图最大边长，默认 `512`
- `--top_frac`: 选择比例，默认 `0.10`
- `--min_per_tissue`: 每个组织最少选择数量，默认 `5`
- `--pca_dim`: PCA 维度，默认 `32`
- `--hsv_bins`: HSV 直方图 bins，默认 `16`
- `--glcm_levels`: GLCM levels，默认 `32`
- `--out_csv`: 结果 CSV
- `--out_failed_csv`: 失败清单 CSV（可选）
- `--cache_dir`: 缩略图缓存目录（可选）

## 4. 输出说明

### `selected_wsi.csv`

- `tissue_type`: 组织类型（目录分组名）
- `tissue_rank`: 该组织内排序（1..k）
- `global_rank`: 全表排序
- `path`: WSI 路径
- `selected_by`: 固定 `kcenter`
- `mean_cosine_distance`: 该样本到所属组织全体样本的平均余弦距离（诊断字段）
- `tissue_ratio`: 组织占比
- `mask_fallback`: 组织 mask 是否 fallback 到整图（0/1）
- `group_total`: 该组织总数
- `group_selected`: 该组织被选数量（`max(ceil(top_frac*N), min_per_tissue)`，并不超过 `N`）

### `failed_wsi.csv`（可选）

- `path`
- `tissue_type`
- `error`

## 5. 流程摘要

1. 缩略图读取（OpenSlide 优先，PIL fallback）。
2. HSV + 形态学组织分割；组织比例过低时 fallback。
3. 提取手工特征：
   - 组织比例
   - RGB/HSV 均值方差
   - HSV 直方图
   - LBP 直方图
   - GLCM 统计
   - 边缘密度与熵
4. 标准化 + PCA。
5. 每个组织独立执行标准化 + PCA + k-center/FPS（cosine distance）选择。

## 6. 退出码

- `0`: 成功
- `2`: 输入为空
- `3`: 全部样本特征提取失败
- `4`: 选择过程异常
