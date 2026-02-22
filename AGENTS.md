# Repository Guidelines

## Project Structure & Modules
- `app.py`: CLI entry; parses prompts, crops with `cli/crop_send.py`, writes `results*.txt/json`.
- `pipeline.py`: Model orchestration (mock vs OpenAI/local), JSON shaping, CLIP evidence, env vars control backend (`USE_MOCK`, `LLM_BACKEND`, `OPENAI_MODEL`, `OPENAI_BASE_URL`).
- `cli/`: chatbot + cropping utilities; keep user-facing CLI helpers here.
- Data: raw in `leaf_disease_vlm/`, curated CSVs `metadata_manifest.csv` and `metadata_clean.csv`, processed crops `processed_vlm_512/`, samples `examples/`.
- Experiments/utilities: `get_topk_evidence.py`, `clip_zero_shot.py`, `few_shots.py`, `preprocess.py`, `make_green_test_image.py`, `script/` helpers. Tests belong in `tests/test_*.py`.

## Build, Test, and Development Commands
- Environment: `python3 -m venv .venv && source .venv/bin/activate && python -m pip install -U pip wheel setuptools && pip install -r requirements.txt`.
- Mocked CLI (offline): `USE_MOCK=1 python app.py --prompt "Classify the leaf" --image examples/leaf.jpg`.
- Real API call: `USE_MOCK=0 OPENAI_API_KEY=... python app.py --prompt "..." --image data/test.jpg`.
- Dataset prep: `python preprocess.py --root leaf_disease_vlm --manifest-out metadata_manifest.csv --clean-out metadata_clean.csv --resize 512 --output-images-dir ./processed_vlm_512 --result-out preprocess_result.json`.
- CLIP check: `python - <<'PY'\nfrom get_topk_evidence import clip_topk_evidence\nprint(clip_topk_evidence("examples/leaf.jpg", k=3))\nPY`.
- Tests: `pytest` in repo root; set `USE_MOCK=1` for stable runs.

## Coding Style & Naming Conventions
- Python 3, PEP 8, 4-space indents, snake_case; CapWords for classes; kebab-case CLI flags.
- Keep functions small; add docstrings to public helpers and note expected env vars. Avoid hardcoded paths; favor repo-relative inputs.
- Logging: keep concise status logs; avoid dumping secrets or full prompts.

## Testing Guidelines
- Use `pytest`; mirror modules with `tests/test_<module>.py`. Prefer unit tests for `pipeline.classify_image`, clipping utilities, and preprocessing defaults using assets in `examples/`.
- For model calls, mock or set `USE_MOCK=1`; for integration snapshots, record sample outputs rather than checking live network responses.
- Add regression tests when changing prompt templates, CLIP evidence ranking, or preprocessing parameters.

## Commit & Pull Request Guidelines
- Commits are short, imperative (e.g., `add mock cache`, `tweak few shots`); keep scope narrow.
- PRs: include summary, key commands run (`pytest`, CLI examples), screenshots/sample outputs for UX/text changes, linked issues/tasks, and backend settings used (`LLM_BACKEND`, `OPENAI_MODEL`, `USE_MOCK`).
- Note any generated artifacts (manifests, processed crops) and whether they are tracked/ignored.

## Security & Configuration Tips
- Store secrets in `.env`; never commit keys. Prefer `USE_MOCK=1` when iterating.
- Large data dirs (`leaf_disease_vlm/`, `processed_vlm_512/`) are often untracked; check `.gitignore` before adding artifacts.
- When switching backends, set `OPENAI_BASE_URL` to the target endpoint and align `LLM_MODEL` with the server.
