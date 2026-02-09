# Repository Guidelines

## Project Structure & Module Organization
- Root script: `select_diverse_wsi.py` (CLI entrypoint and full pipeline: discovery, thumbnail loading, feature extraction, PCA, k-center selection, CSV output).
- Dependency manifest: `requirements.txt`.
- User-facing usage notes: `README.md`.
- Product requirements/spec: `WSI_kcenter_diversity_selection_requirements.md`.
- Recommended future layout for growth:
  - `tests/` for automated tests
  - `samples/` for tiny non-sensitive demo images
  - `outputs/` for generated CSV artifacts (gitignored)

## Build, Test, and Development Commands
- Create environment and install deps:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Run selection from directory input:
  - `python3 select_diverse_wsi.py --input_dir /data/wsi --extensions "svs,tif,tiff" --top_frac 0.1 --min_per_tissue 5 --out_csv selected_wsi.csv`
- Quick syntax check:
  - `python3 -m py_compile select_diverse_wsi.py`

## Coding Style & Naming Conventions
- Language: Python 3; follow PEP 8.
- Indentation: 4 spaces; keep lines readable and functions focused.
- Naming:
  - `snake_case` for functions/variables
  - `UPPER_SNAKE_CASE` for constants
  - meaningful CLI flags (e.g., `--thumb_side`, `--top_frac`)
- Prefer small, testable helpers over monolithic blocks.

## Testing Guidelines
- Current status: no formal test suite committed yet.
- Minimum requirement before merging:
  - run `python3 -m py_compile select_diverse_wsi.py`
  - run one smoke test on at least 3 files and verify `selected_wsi.csv` is produced.
- When adding tests, use `pytest` with files named `tests/test_*.py`.

## Commit & Pull Request Guidelines
- Repository currently has no commit history; adopt Conventional Commits:
  - `feat: add tif fallback for thumbnail loading`
  - `fix: guard pca_dim for small datasets`
- PRs should include:
  - concise summary of behavior changes
  - example command used for verification
  - output evidence (CSV column preview or log snippet)
  - linked issue/task if available.

## Security & Data Handling
- Do not commit raw WSI data, patient identifiers, or large generated caches.
- Keep local caches (`thumb_cache/`, output CSVs) out of version control unless explicitly required.
