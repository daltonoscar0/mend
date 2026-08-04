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
- 2026-08-04: WikiText is whitespace-tokenised, so its punctuation arrives as standalone tokens; without a detokenisation pass the clean targets lost every comma and full stop and would have taught the model to strip punctuation. Added `detokenize()` and a filter that rejects any sentence whose tokens do not survive intact.
- 2026-08-04: WikiText is streamed rather than downloaded in full, and only the first N well-formed sentences are kept, which keeps generation at ~15 s instead of pulling the whole 180 MB corpus.
- 2026-08-04: The streaming reader leaves a live worker thread behind that stalls interpreter shutdown, so `generate_data` exits via `os._exit` once the files are flushed.
- 2026-08-04: Source pool is 65% WikiText / 35% templates, and each source sentence is injected into at most two pairs (different seeds), so 54k pairs need ~27k distinct clean sentences.
- 2026-08-04: 6 of the 30 imperative templates (20%) are withheld from training entirely and appear only in the test split, tagged `template_heldout`, so unseen-frame generalisation can be read off separately.
- 2026-08-04: About a third of pairs carry no disfluency at all, so the model has to learn that leaving text alone (beyond punctuation restoration) is often correct.
- 2026-08-04: Slot values for `topic` were bare nouns after templates were fixed; the first draft produced "the the budget review" because both the template and the slot supplied an article.
