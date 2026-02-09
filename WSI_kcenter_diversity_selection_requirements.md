# WSI 数据集多样性优先标注筛选（k-center）需求文档（Codex 友好版）

> 目标：在 给定的WSI 数据集（主文件夹及代表不同组织的子文件夹）中筛选 **多样性最高（覆盖面最大）** 的前 10% WSI，用于优先标注，提升标注效率。  
> 策略：**k-center / Farthest Point Sampling（FPS）** 在低维嵌入空间中做“覆盖面最大化”的代表性采样。  
> 约束：**计算资源友好、快速实现、无需大模型/Transformer 特征**，面向 `.svs` 与 `.tif`。

---

## 1. 背景与问题定义

### 1.1 背景
- 数据类型：WSI（Whole Slide Image）全切片图像，主要格式：`svs`、`tif`。
- 痛点：WSI 标注成本极高（时间、人力），需要优先标注最能覆盖数据集多样性的样本。
- 目标：选出一个子集（默认 10%），使其尽量覆盖整个数据分布（减少重复标注、增加多样性、每种组织单独计算）。

### 1.2 关键概念
- **相似度最低 Top-K**：偏“离群/异常”优先。
- **覆盖面最大（k-center）Top-K**：偏“代表性/覆盖性”优先（本项目采用）。

---

## 2. 范围（Scope）

### 2.1 In Scope
- 从输入目录（或文件列表）读取 WSI（svs/tif）。
- 为每张 WSI 生成低倍率缩略图（thumbnail）。
- 提取**低维、资源友好**的手工特征（颜色 + 粗纹理 + 结构统计）。
- 标准化 + PCA 降维得到低维嵌入。
- 在嵌入空间执行 **k-center / FPS** 选择 `ceil(0.1 * N)` 张 WSI。
- 输出：
  - 结果 CSV（路径、rank、可选评分字段）
  - 运行日志（可选：每张图处理耗时、失败列表）
  - 可选：每张 WSI 的缩略图缓存

### 2.2 Out of Scope（明确不做）
- 使用深度网络/Transformer 进行特征提取
- 像素级全分辨率比对
- 复杂的跨节点分布式计算/大规模 GPU 推理

---

## 3. 约束与非功能需求（NFR）

### 3.1 计算资源约束
- **默认 CPU-only** 可运行
- 每张 WSI 只处理缩略图（最大边长默认 512 或 768），避免读取全分辨率
- 特征维度：~80–120（可调），PCA 后嵌入维度默认 32

### 3.2 快速实现约束
- 依赖：`numpy`, `opencv-python`, `scikit-image`, `scikit-learn`, `pandas`, `Pillow`, `tqdm`
- WSI 读取：优先 `openslide`（svs），tif 可用 `openslide` 或 `tifffile`（按环境选择）

### 3.3 稳健性
- 支持组织区域很少、背景很多的 WSI（组织 mask 失败需 fallback）
- 对染色差异/扫描差异不过度敏感（允许一定影响，但需提供可调参数）

---

## 4. 输入 / 输出规范

### 4.1 输入
- 输入方式 A：`--input_dir` + `--pattern`（如 `*.svs`、`*.tif`）
- 输入方式 B：显式文件清单（text/csv）`--input_list`

### 4.2 输出
- CSV：`selected_wsi.csv`（字段建议）
  - `rank`: 1..k
  - `path`: 文件绝对/相对路径
  - `selected_by`: 固定值 `kcenter`
  - `optional_metrics`（可选）：例如 mean cosine similarity（仅用于解释/诊断）
- 可选：`failed_wsi.csv`（失败文件路径 + 错误信息）
- 可选：thumbnail 缓存目录 `thumb_cache/`（便于复跑加速）

---

## 5. 处理流程（Pipeline）

### 5.1 缩略图生成（关键性能点）
- 目标：为每张 WSI 得到 RGB thumbnail（H×W×3，uint8）
- 要求：
  - 保持长宽比
  - 最大边长 `thumb_side` 可配置（默认 512；<100 张可用 768）
  - 对 `.svs`：优先用 OpenSlide 的 `get_thumbnail`
  - 对 `.tif`：优先读取低分辨率层（若存在），否则整体缩放

### 5.2 组织区域 mask（建议启用）
- 目的：避免白背景主导颜色/纹理特征
- 推荐做法（快速、实现简单）：
  - HSV 空间：用 S 通道 Otsu 阈值找组织；用 V 通道剔除高亮背景
  - 形态学开闭操作去噪
- Fallback：
  - 若组织比例 < 1%：认为 mask 失败，使用整图计算特征，并记录告警

### 5.3 手工特征（低维、可解释）
> 特征应当“足够区分”，但不追求语义最优；优先速度与稳健。

建议特征组：
1. **组织比例**：tissue_ratio（1 维）
2. **颜色统计**（组织内）：RGB/HSV 的 mean/std（12 维）
3. **颜色分布**：HSV 直方图（每通道 16 bins → 48 维；可降到 8 bins）
4. **纹理**：
   - LBP（uniform）直方图（10 维）
   - GLCM 若干统计量（例如 contrast/homogeneity/energy/correlation 的 mean/std，约 12 维）
5. **结构粗特征**：
   - 边缘密度（Canny，1 维）
   - 熵（entropy，1 维）

> 备注：特征总维度约 80–120。维度越大不一定更好，需与 PCA 配合。

### 5.4 标准化与降维
- `StandardScaler`：各维归一化到零均值单位方差
- `PCA(n_components=pca_dim)`：默认 32，范围建议 16–64
- 输出：嵌入矩阵 `X`（N×pca_dim）

### 5.5 多样性选择：k-center / FPS
- 目标：从 N 个点中选 k 个，使得每个未选点到“已选集合”的最小距离尽量大（覆盖面最大）
- 距离度量：推荐 **cosine distance = 1 - cosine similarity**
- 初始化点建议：
  - 选择“全局平均相似度最低”的点作为第一个中心（比纯随机更稳）
- 贪心迭代：
  - 维护每个点到已选集合的最小距离 `min_dist`
  - 每次选择 `argmax(min_dist)` 作为下一个中心
  - 更新 `min_dist = min(min_dist, dist_to_new_center)`

---

## 6. 参数与默认值（建议）

### 6.1 通用默认
- `top_frac = 0.10`
- `thumb_side = 512`
- `pca_dim = 32`
- `hsv_bins = 16`
- `glcm_levels = 32`
- `lbp_P=8, R=1 (uniform)`

### 6.2 不同规模建议
- N < 100：
  - `thumb_side=768`（更稳）
  - `pca_dim=16~32`
- 100 ≤ N ≤ 1000：
  - `thumb_side=512`
  - `pca_dim=32`
- N > 1000（如后续扩展）：
  - `thumb_side=256~384`
  - `hsv_bins=8`
  - `pca_dim=16~32`
  - 可选：先 mini-batch 聚类再做簇内代表性采样（非必须）

---

## 7. CLI / 接口规范（Codex 友好）

### 7.1 命令行入口
- 脚本名建议：`select_diverse_wsi.py`

### 7.2 关键参数（必须支持）
- `--input_dir`：WSI 目录
- `--pattern`：glob pattern（如 `*.svs` 或 `*.tif`）
- `--input_list`：可选，显式文件列表（优先级高于 input_dir）
- `--thumb_side`
- `--top_frac`
- `--pca_dim`
- `--out_csv`
- `--out_failed_csv`（可选）
- `--cache_dir`（可选）

### 7.3 返回码
- `0`：成功且输出 CSV
- `非0`：致命错误（如无文件、提取失败过多等）

### 7.4 日志规范
- 每张 WSI 处理：`[OK] path=... time=... tissue_ratio=...`
- 失败：`[FAIL] path=... err=...`
- 总结：N、k、失败数、总耗时、均耗时

---

## 8. 验收标准（Acceptance Criteria）

### 8.1 功能正确性
- 输入 N 张 WSI，输出 k=ceil(0.1*N) 条记录（除非失败导致可用数不足）
- 输出 CSV 字段完整且路径可用
- k-center 选择结果可复现（固定随机种子或确定性初始化）

### 8.2 性能与资源
- 处理每张 WSI 仅依赖缩略图（无全分辨率读入）
- 在常规工作站 CPU 上，N=100~1000 可在合理时间完成（以缩略图读取速度为主）
- 内存占用：特征矩阵（N×~100）与嵌入矩阵（N×32）为主，典型 < 数百 MB

### 8.3 结果“多样性”可解释性（最低要求）
- 输出附带可选诊断指标（例如 mean cosine similarity）便于人工 spot-check
- 可选生成“被选样本缩略图拼图/列表”（非必须，但利于人工验证）

---

## 9. 风险点与缓解

### 9.1 染色/扫描批次主导多样性
- 风险：算法可能偏向挑选“颜色差异大”的 WSI
- 缓解：
  - 增加对组织纹理的权重（LBP/GLCM）
  - 降低颜色直方图维度（bins=8）或在 PCA 前对颜色特征做缩放权重
  - 可选：简单 stain normalization（非必须，慎用以保持快速实现）

### 9.2 组织 mask 失败
- 风险：组织极少或异常背景导致 tissue mask 不稳
- 缓解：
  - tissue_ratio < 1% 时 fallback 全图
  - 记录告警并在结果里标记

### 9.3 `.tif` 兼容性
- 风险：多页 TIFF / pyramidal TIFF 读取差异
- 缓解：
  - 优先 openslide 能读则用 openslide
  - 读取失败则 fallback PIL；必要时引入 `tifffile`（作为可选依赖）

---

## 10. 交付物清单（Deliverables）

1. `select_diverse_wsi.py`（CLI 可执行）
2. `requirements.txt` 或 `environment.yml`（依赖声明）
3. `README.md`（快速使用说明）
4. 示例输出：
   - `selected_wsi.csv`
   - `failed_wsi.csv`（如有）
   - （可选）`thumb_cache/`

---

## 11. 实施任务拆分（Codex 执行建议）

- Task A：文件发现与输入解析（dir/pattern/list）
- Task B：缩略图读取模块（openslide + fallback）
- Task C：tissue mask 模块（HSV + morph + fallback）
- Task D：特征提取模块（color/hist/texture/structure）
- Task E：标准化 + PCA 模块
- Task F：k-center/FPS 选择模块（cosine distance）
- Task G：输出与日志（CSV + 失败清单）
- Task H：最小测试集 smoke test（>=3 张 WSI）

---

## 12. 依赖与安装（建议写入 README）

最小依赖：
- numpy, pandas, pillow, opencv-python, scikit-image, scikit-learn, tqdm

WSI 支持（推荐）：
- openslide-python + openslide（系统库或 conda-forge）

---

### 版本信息
- 文档生成日期：2026-02-09
