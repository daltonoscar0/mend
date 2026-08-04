# Decisions

A running log of engineering decisions made while building Mend, one line each.

- 2026-08-04: Target hardware is an Apple Silicon Mac (M-series, 16 GB, no CUDA), so all "GPU" paths use MPS and CPU is the fallback.
- 2026-08-04: Used `uv` with a Python 3.12 virtualenv at `.venv` rather than the system Python 3.9, so current `transformers`/`datasets` releases install cleanly.
- 2026-08-04: Pinned dependencies to minor versions in `pyproject.toml` (torch 2.9, transformers 4.57, datasets 4.4, peft 0.18) for reproducibility.
