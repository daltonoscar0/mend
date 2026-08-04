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
- 2026-08-04: Full fine-tune rather than LoRA: flan-t5-small is 77M parameters and fits comfortably in unified memory, so there was no memory reason to reach for adapters. A `--lora` flag is kept for a tighter machine.
- 2026-08-04: Measured MPS throughput at batch 64 was pathological (19 s/step, degrading) while batch 32 held ~1-3 s/step, so batch 32 is the default. Step time tracks sequence length rather than batch size, so `group_by_length=True` is on.
- 2026-08-04: Cut the run from 3 epochs to 2 (3126 steps, ~2 h at the measured ~2.3 s/step). Per the brief, epochs come off before the model gets bigger; the 200-step smoke had already reached eval loss 0.29, so the task is not data-starved.
- 2026-08-04: Evaluation decodes greedily (no beam search), matching the smoke run and keeping the eval cheap and deterministic.
- 2026-08-04: Exact match normalises whitespace and treats the sentence-final mark as optional, but is otherwise case-sensitive, since restoring capitalisation is part of the task.
- 2026-08-04: WER is computed over whitespace tokens with punctuation still attached, so a missing comma costs one substitution; punctuation restoration is part of what is being measured.
- 2026-08-04: Filler removal F1 counts occurrences rather than positions, because a filler word can also be a real word in the same sentence. Filler-vocabulary tokens that some other disfluency type deletes (a false start beginning "can you") are excluded from the row, otherwise a perfect output scores precision 0.97 instead of 1.00.
- 2026-08-04: The hard set has no span annotations, so its filler and repair metrics come from the input/gold word-multiset difference. On the synthetic split, where both are available, the two definitions agree exactly.
- 2026-08-04: Surprisal is summed over the subwords of a word, and the causal LM's offsets fold the preceding space into each token, which shifted every measurement one word to the left until it was corrected.
- 2026-08-04: The surprisal control is the same sentence with its disfluencies removed, measured at the matching word position and differenced instance by instance, rather than a separate pool of fluent sentences.
- 2026-08-04: Only disfluencies with a complete +/-5 window on both the disfluent and the clean side are analysed, so every point on the figure rests on the same set of instances.
- 2026-08-04: The first full training run degraded from 1.8 s/step to 27 s/step and drove the machine 7.9 GB into swap. Cause: the MPS caching allocator keeps a pool per tensor shape and never releases it, so dynamically padded batches grew the pool all run. Fixed by padding to a multiple of 8 and emptying the MPS cache every 100 steps; step time is now flat at ~1.4 s.
- 2026-08-04: Training reads only the `disfluent`/`clean` fields from the JSONL rather than the full annotated records, which are only needed by the evaluation and surprisal stages.
- 2026-08-04: Do not run the surprisal pass alongside training on this machine; the two together exhaust physical memory and both crawl.
- 2026-08-04: Repair resolution reached 94.5% on the synthetic dev/test split, comfortably above the 60% bar, so flan-t5-small stands and there was no reason to escalate to flan-t5-base.
- 2026-08-04: Early stopping never fired: validation loss fell at every one of the seven evaluations (0.178 -> 0.131), so the run used both epochs in full.
- 2026-08-04: The surprisal analysis was run and committed before the evaluation, out of milestone order, because it needs no trained model and could use the CPU while the GPU was busy.
- 2026-08-04: `outputs/predictions.jsonl` is committed alongside the table so every cell can be traced back to the decoded output it came from.
