# Relative Path History Migration Design

## Goal

Update the WSI selection pipeline so that:

1. `selected_wsi.csv` and `failed_wsi.csv` store `path` as a path relative to `--input_dir`, not an absolute filesystem path.
2. Existing history files that still store absolute paths are migrated in place to the new relative-path format at startup.
3. Different devices can select the same slides for the same round when they use the same dataset layout and the same `--seed`.

This design keeps the current feature extraction and per-tissue k-center selection flow intact and only changes path identity, history loading, and reproducibility safeguards.

## Confirmed Decisions

- Keep `--seed` as a CLI parameter only.
- Do not add a `seed` column to any CSV output.
- Automatically migrate legacy absolute-path history files to relative paths.
- Treat relative path under `--input_dir` as the only supported history identity after migration.
- Fail fast if a legacy absolute path cannot be mapped under the current `--input_dir`.

## Design

### 1. Path Identity

Discovery and thumbnail loading continue to use absolute filesystem paths because image IO depends on real files. CSV output and history matching switch to a normalized relative path derived from `Path(path).resolve()` relative to `Path(input_dir).resolve()`.

This relative path becomes the canonical sample key for:

- `selected_wsi.csv`
- `failed_wsi.csv`
- history de-duplication
- incremental round filtering

The design assumes that dataset-internal layout is stable across devices. If two users mount the same dataset in different absolute locations but preserve the same directory tree under `input_dir`, they will produce the same CSV `path` values.

### 2. Legacy History Migration

When loading `selected_wsi.csv`, the script inspects the `path` column:

- if rows are already relative, keep them unchanged
- if rows are absolute, convert each row to a relative path under the current `input_dir`
- if any absolute path falls outside the current `input_dir`, raise a clear error and stop

After successful conversion, rewrite `selected_wsi.csv` in place before the rest of the run proceeds. This avoids mixed history formats and keeps later rounds simple and deterministic.

This migration is explicit and non-silent on failure because history is the source of truth for incremental selection. A wrong migration would corrupt future rounds.

### 3. Deterministic Selection Across Devices

The current pipeline is already close to deterministic, but this change makes the ordering guarantees explicit:

- sort discovered WSI candidates deterministically
- group by tissue type using stable sorted tissue names
- preserve deterministic row order inside each tissue by sorting on canonical relative path
- continue passing `--seed` into PCA
- prefer an explicitly deterministic PCA solver configuration to avoid environment-dependent solver selection

The selection algorithm itself is deterministic once the feature matrix row order is deterministic. Because k-center/FPS starts from the sample with the maximum mean cosine distance and then greedily selects the farthest remaining point, consistent ordering and consistent PCA output are the key requirements.

## Implementation Scope

### Files to Modify

- `select_diverse_wsi.py`
- `README.md`
- `docs/DEMANDS.MD`
- `docs/CHANGELOG.md`

### Code Changes

Add small helper functions rather than restructuring the pipeline:

- convert absolute WSI path to canonical relative path
- detect whether a CSV path is absolute or relative
- migrate history dataframe paths to relative format
- build a normalized relative-path set for history exclusion

Then update row creation so both success rows and failure rows write relative `path` values.

## Error Handling

The design intentionally avoids fallback behavior that could hide a history mismatch.

The script should raise an error when:

- `selected_wsi.csv` exists but lacks a `path` column
- a legacy absolute history path does not belong to the current `input_dir`
- a discovered WSI path cannot be expressed relative to `input_dir`

The script should not silently skip malformed history rows or mixed-dataset entries.

## Verification

Minimum verification for implementation:

1. Legacy migration test
   - seed `selected_wsi.csv` with absolute paths under `input_dir`
   - run the script
   - verify the file is rewritten with relative paths only

2. Cross-device reproducibility test
   - place the same dataset tree under two different absolute root directories
   - run the same command with the same `--seed`
   - verify the selected relative paths and ranks are identical

3. Failure boundary test
   - seed `selected_wsi.csv` with at least one absolute path outside `input_dir`
   - verify the script stops with a clear error

4. Regression check
   - verify incremental rounds still exclude prior selections and append the next round correctly
