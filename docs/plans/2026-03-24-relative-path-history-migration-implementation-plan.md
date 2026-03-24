# Relative Path History Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Change CSV `path` fields to be relative to `--input_dir`, migrate legacy absolute-path history in place, and keep same-round selection deterministic across devices when the dataset layout and seed match.

**Architecture:** Keep the existing thumbnail, feature extraction, and k-center selection pipeline unchanged. Introduce a small path-normalization layer around discovery/history/output, add deterministic ordering safeguards before per-tissue selection, and update docs plus versioned release notes to reflect the CSV format change.

**Tech Stack:** Python 3, pandas, numpy, scikit-learn, unittest

---

### Task 1: Capture the new behavior with failing tests

**Files:**
- Modify: `tests/test_incremental_round_selection.py`

**Step 1: Write the failing tests**

Add tests that assert:
- new `selected_wsi.csv` rows store paths relative to `input_dir`
- legacy absolute-path history is rewritten in place to relative paths before selection continues
- legacy history rows outside `input_dir` raise a clear error

**Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests/test_incremental_round_selection.py -v`
Expected: FAIL because the current implementation writes absolute paths and does not migrate legacy history.

### Task 2: Implement path canonicalization and history migration

**Files:**
- Modify: `select_diverse_wsi.py`

**Step 1: Add relative-path helpers**

Add helpers that:
- normalize `input_dir`
- convert discovered absolute WSI paths to canonical relative paths
- normalize history rows from either relative or absolute path format

**Step 2: Migrate legacy history in place**

Load `selected_wsi.csv`, detect absolute-path rows, convert them relative to `input_dir`, and rewrite the file before continuing. Raise an error if any legacy row is outside `input_dir`.

**Step 3: Switch history filtering and CSV output to relative paths**

Keep real file access on absolute paths, but make history exclusion, `selected_wsi.csv`, and `failed_wsi.csv` all use canonical relative paths.

**Step 4: Tighten determinism**

Ensure candidate ordering is stable by sorting tissues and per-tissue rows on canonical relative path before feature matrix construction and selection. Make PCA use an explicitly deterministic solver configuration.

### Task 3: Update repository metadata and user docs

**Files:**
- Create: `docs/DEMANDS.MD`
- Create: `docs/CHANGELOG.md`
- Modify: `README.md`
- Modify: `select_diverse_wsi.py`

**Step 1: Record the structured demand**

Document the confirmed requirements for relative-path output, history migration, and deterministic same-round selection.

**Step 2: Bump the version**

Assign a new semantic version and update any version-bearing code or docs consistently.

**Step 3: Update user-facing documentation**

Document that:
- CSV `path` is relative to `input_dir`
- old history files are auto-migrated
- same dataset layout plus same seed produces the same selected slides across devices

### Task 4: Verify end to end

**Files:**
- Test: `tests/test_incremental_round_selection.py`

**Step 1: Run unit tests**

Run: `python3 -m unittest tests/test_incremental_round_selection.py -v`
Expected: PASS

**Step 2: Run syntax/version checks**

Run: `python3 -m py_compile select_diverse_wsi.py`
Expected: PASS

Run: `python3 select_diverse_wsi.py --version`
Expected: exit `0` and print the bumped script version
