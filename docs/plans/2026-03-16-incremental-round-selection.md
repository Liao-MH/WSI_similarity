# Incremental Round Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make repeated runs exclude WSI already listed in `output/selected_wsi.csv`, append the newly selected WSI as the next round, and report round/remaining counts at the end.

**Architecture:** Treat `output/selected_wsi.csv` as the single source of truth for prior selections. Load prior rows before discovery-based selection, filter the candidate set by prior `path` values, compute the next round number, and append this round's rows while preserving old history. Keep the existing feature extraction and per-tissue k-center pipeline unchanged after candidate filtering.

**Tech Stack:** Python 3, pandas, numpy, scikit-learn, unittest

---

### Task 1: Lock down the new behavior with a regression test

**Files:**
- Create: `tests/test_incremental_round_selection.py`
- Modify: `select_diverse_wsi.py`

**Step 1: Write the failing test**

Write a test that seeds an existing `output/selected_wsi.csv`, runs `run(args)`, and asserts:
- previously selected paths are excluded from the new round
- appended rows contain `round = 2`
- `global_rank` restarts from `1` within the new round
- the summary message reports round count and the "已全部挑选" suffix when remaining samples are fewer than the per-round target

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_incremental_round_selection.py -v`
Expected: FAIL because `run(args)` currently overwrites `selected_wsi.csv`, does not append a `round` column, and does not print the new summary message.

### Task 2: Implement history-aware round selection

**Files:**
- Modify: `select_diverse_wsi.py`

**Step 1: Add helpers to load prior selections**

Read `args.out_csv` if it exists, normalize legacy files without `round` by treating old rows as `round = 1`, and return:
- historical rows
- selected path set
- next round number

**Step 2: Filter candidates before feature extraction**

Exclude previously selected `path` values from the discovered WSI list. If no remaining paths exist, write no new rows and print the round summary with zero newly selected samples.

**Step 3: Append this round's rows**

Annotate every new row with `round`, compute `global_rank` within the new round only, concatenate historical rows plus new rows, and write the merged dataframe back to `args.out_csv`.

**Step 4: Emit the requested summary**

Print:
- `经过 N 轮挑选，共选过 X 张 WSI，本轮还剩 R 张 WSI，已选择 S 张`
- append `因剩余数量'R'小于每轮挑选设定数量'T'，已全部挑选` when `R < T`

### Task 3: Update docs and verify

**Files:**
- Modify: `README.md`

**Step 1: Document incremental behavior**

Update usage/output sections to explain that:
- reruns skip paths already listed in `output/selected_wsi.csv`
- `selected_wsi.csv` is append-only history
- `round` indicates which run selected each WSI

**Step 2: Run verification**

Run: `python3 -m unittest tests/test_incremental_round_selection.py -v`
Expected: PASS

Run: `python3 select_diverse_wsi.py --version`
Expected: exit `0` and print the script version
