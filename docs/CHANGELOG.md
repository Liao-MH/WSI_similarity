v2.0.0 - 2026-03-24
用户需求
用户要求把 `selected_wsi.csv` 与 `failed_wsi.csv` 的 `path` 字段从绝对路径改为基于 `input_dir` 的相对路径，并让不同设备在相同轮次、相同 seed 下选到相同的 slide；同时旧版历史 CSV 需要自动迁移而不是手动处理。本轮还要求取消 `--version` 命令行参数。
已做改动
版本号升级到 v2.0.0
`select_diverse_wsi.py`
新增相对路径规范化与旧历史 CSV 原地迁移逻辑
将历史去重、结果输出和失败输出统一切换为相对 `input_dir` 的路径
按相对路径稳定排序候选样本，并将 PCA 切换为确定性求解配置
移除 `--version` 命令行参数
`tests/test_incremental_round_selection.py`
新增对相对路径输出、旧历史迁移成功、旧历史越界报错、失败 CSV 相对路径输出、已移除 `--version` 参数的测试
`README.md`
更新版本说明、CSV 字段语义、旧历史自动迁移说明、跨设备复现说明和 `--version` 参数移除说明
`docs/DEMANDS.MD`
记录本轮结构化需求
影响文件
select_diverse_wsi.py
tests/test_incremental_round_selection.py
README.md
docs/DEMANDS.MD
docs/CHANGELOG.md
docs/plans/2026-03-24-relative-path-history-migration-design.md
docs/plans/2026-03-24-relative-path-history-migration-implementation-plan.md
验证结果
python3 -m unittest tests/test_incremental_round_selection.py -v
python3 -m py_compile select_diverse_wsi.py
