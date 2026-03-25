# WSI Diversity Selection (k-center/FPS)

基于缩略图和手工特征（颜色 + 纹理 + 结构）对 WSI 做多样性优先筛选，输出覆盖面最大的 Top-K（默认 10%），用于优先标注。

当前版本：`v2.0.1`

## 1. 安装

### 方式 A：`venv`（已有方式）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 方式 B：`conda`（推荐用于 WSI/OpenSlide）

```bash
conda create -n wsi_similarity python=3.10 -y
conda activate wsi_similarity
conda install -c conda-forge openslide openslide-python -y
pip install -r requirements.txt
```

或使用仓库内环境文件一键创建：

```bash
conda env create -f environment.yml
conda activate wsi_similarity
```

`svs` 读取建议安装 OpenSlide 系统库（`openslide-python` 仅为 Python 绑定）。若环境无 OpenSlide，脚本会自动尝试用 PIL fallback（适用于多数 tif）。

## 2. 用法

```bash
python3 select_diverse_wsi.py \
  --input_dir /path/to/wsi_root \
  --extensions "svs,tif,tiff" \
  --output_dir output \
  --thumb_side 512 \
  --top_frac 0.10 \
  --min_per_tissue 5 \
  --pca_dim 32 \
  --out_csv selected_wsi.csv \
  --out_failed_csv failed_wsi.csv \
  --cache_dir thumb_cache
```

脚本会递归扫描 `--input_dir` 下所有匹配后缀文件，并按一级子目录作为组织类型分组（例如 `Breast cancer N=137`）。

若使用 conda，请先激活环境再运行：

```bash
conda activate wsi_similarity
python3 select_diverse_wsi.py --input_dir /path/to/wsi_root --out_csv selected_wsi.csv
```

所有输出会统一写入 `output/`（可通过 `--output_dir` 修改）。若目录不存在会自动创建。

若 `output/selected_wsi.csv` 已存在，脚本会先读取其中历史已选的 WSI 路径，并在新一轮运行时自动排除这些样本；本轮新结果会继续追加到同一个 `selected_wsi.csv` 中，而不是覆盖旧结果。

当前实现中，`selected_wsi.csv` 与 `failed_wsi.csv` 中的 `path` 均保存为“相对于 `--input_dir` 的相对路径”，以避免不同设备上的绝对路径差异影响续跑与对比。

当前实现中，脚本不再提供 `--version` 命令行参数。

若检测到旧版 `selected_wsi.csv` 中仍为绝对路径，脚本会在启动时自动将其迁移为相对路径后再继续运行；如果某条旧路径不属于当前 `--input_dir`，程序会直接报错并停止，而不会静默跳过。

在数据集目录结构一致、`--seed` 一致、轮次一致的前提下，不同设备应得到相同的入选 slide。

## 3. 关键参数

- `--output_dir`: 输出目录，默认 `output`
- `--input_dir`: WSI 目录
- `--extensions`: 自动识别后缀（逗号分隔），默认 `svs,tif,tiff`
- `--thumb_side`: 缩略图最大边长，默认 `512`
- `--top_frac`: 选择比例，默认 `0.10`
- `--min_per_tissue`: 每个组织最少选择数量，默认 `5`
- `--pca_dim`: PCA 维度，默认 `32`
- `--hsv_bins`: HSV 直方图 bins，默认 `16`
- `--glcm_levels`: GLCM levels，默认 `32`
- `--out_csv`: 结果 CSV 文件名（保存到 `output_dir`）
- `--out_failed_csv`: 失败清单 CSV 文件名（保存到 `output_dir`）
- `--cache_dir`: 缩略图缓存子目录名（创建在 `output_dir` 下）

## 4. 输出说明

### `selected_wsi.csv`

- `round`: 第几轮被选中（首次运行是 `1`，之后每次运行依次递增）
- `tissue_type`: 组织类型（目录分组名）
- `tissue_rank`: 该轮内、该组织内排序（1..k）
- `global_rank`: 该轮内全表排序
- `path`: 相对于 `--input_dir` 的 WSI 路径
- `selected_by`: 固定 `kcenter`
- `mean_cosine_distance`: 该样本到所属组织全体样本的平均余弦距离（诊断字段）
- `tissue_ratio`: 组织占比
- `mask_fallback`: 组织 mask 是否 fallback 到整图（0/1）
- `group_total`: 该组织总数
- `group_selected`: 该组织被选数量（`max(ceil(top_frac*N), min_per_tissue)`，并不超过 `N`）

`selected_wsi.csv` 为累计历史账本：每次运行只会从“尚未出现在该 CSV 中”的 WSI 里继续挑选，并将新一轮结果追加到文件末尾。若该文件中仍存在旧版绝对路径，脚本会先自动迁移成相对路径。

### `failed_wsi.csv`（可选）

- `path`：相对于 `--input_dir` 的 WSI 路径
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
6. 若已存在历史 `selected_wsi.csv`，则先将其中旧版绝对路径迁移为相对路径，再排除历史已选样本，对剩余样本执行本轮选择，并在结束时输出轮次摘要。
7. 为保证跨设备复现，候选样本和组内样本会按相对路径稳定排序，PCA 使用确定性配置。

## 6. 退出码

- `0`: 成功
- `2`: 输入为空
- `3`: 全部样本特征提取失败
- `4`: 选择过程异常
