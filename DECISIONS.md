# Decisions

A running log of engineering decisions made while building Mend, one line each.

- 2026-08-04: Target hardware is an Apple Silicon Mac (M-series, 16 GB, no CUDA), so all "GPU" paths use MPS and CPU is the fallback.
- 2026-08-04: Used `uv` with a Python 3.12 virtualenv at `.venv` rather than the system Python 3.9, so current `transformers`/`datasets` releases install cleanly.
- 2026-08-04: Pinned dependencies to minor versions in `pyproject.toml` (torch 2.9, transformers 4.57, datasets 4.4, peft 0.18) for reproducibility.
- 2026-08-04: The injector plans all edits over clean-token positions first and reserves overlapping positions, then emits left to right, so disfluent-side span offsets are exact by construction instead of being patched up afterwards.
- 2026-08-04: `InjectionResult` carries a token-level `alignment` (disfluent index -> clean index, `None` for injected material), which makes the round-trip test exact rather than case-insensitive and gives the evaluator a way to score filler deletion per token.
- 2026-08-04: Clean tokens with no alphanumeric content (stray em dashes) have no spoken form, so they are dropped and `clean_text` is rebuilt without them; this keeps the round trip exact.
- 2026-08-04: Repairs are restricted to words that fall into a closed swap group (names, days, months, numerals, colours, rooms, document types) or bare digits, so the reparandum is always plausible; sentences with no swappable word simply get no repair.
- 2026-08-04: Repair reparandum spans are one token. Multi-token repairs are left to the hand-written hard set rather than the injector, to keep span bookkeeping simple.
- 2026-08-04: False-start fragments are drawn from a fixed list of clause openers rather than truncating a real word, since sub-word fragments ("the comm—") would not survive the whitespace tokenisation the spans are defined over.
- 2026-08-04: Punctuation degradation is unconditional (always lowercase, always strip edge punctuation) rather than probabilistic, so the model sees a consistent input distribution; internal apostrophes and hyphens are kept because ASR emits them.
