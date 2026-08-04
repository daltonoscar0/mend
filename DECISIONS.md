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
- 2026-08-04: Training logs are untracked (`outputs/*.log`); they are mostly progress-bar carriage returns. The numbers worth keeping live in `train_summary.json`, `eval_table.*` and `surprisal_effects.json`.
- 2026-08-04: Added `jiwer` to the dependency list (not in the original set) rather than hand-rolling an edit-distance WER; pytest sits in an optional `dev` extra, so the README's setup line installs `.[dev]`.

## Second pass: injector coverage, a larger test split, and real speech

- 2026-08-04: Repairs now cover spans of up to three tokens, built either by substituting a word inside the span or by dropping one, so the reparandum and the repair share most of their words. That is the "the report ... the quarterly report" case the single-token version could never produce.
- 2026-08-04: Added delayed repairs, where the speaker carries on for a few words before circling back to correct. The correction therefore arrives after the words it belongs in front of, which is what a cross-clause repair looks like.
- 2026-08-04: A delayed repair breaks the assumption that spoken order matches written order, so `remove_disfluencies` now reassembles by clean-token index rather than by position, and the alignment test asks for a permutation rather than a monotone run.
- 2026-08-04: A delayed correction always gets an editing phrase; without one there is nothing to distinguish "send it to Sarah before Friday, no, report" from a list.
- 2026-08-04: False starts can now happen at a later clause boundary rather than only at the start of a sentence.
- 2026-08-04: Fillers are never injected next to a literal use of the same word. "like like-minded people" is unresolvable for a reader too, so it is noise rather than signal.
- 2026-08-04: 30% of the WikiText pool is now reserved for sentences that use a filler-like word literally, and eight dictation templates were added where "like", "so", "well", "no", "right", "sorry", "wait" and "know" do real work. Deleting every instance of "like" is no longer a winning strategy.
- 2026-08-04: The new templates were checked against the hard set: zero generated sentences match a hard-set gold.
- 2026-08-04: **The filler-as-content and repair-shape fixes were made in response to hard-set failures, so the hard set is now a development set for those categories rather than a clean held-out measurement.** SwDA takes over as the untouched benchmark.
- 2026-08-04: Test split grown from 2k to 10k pairs, which takes the surprisal analysis from 64 usable repair windows to several hundred.
- 2026-08-04: Real speech comes from the Switchboard Dialog Act corpus, which is freely redistributable (no LDC licence) and whose transcripts carry the Treebank-3 disfluency markup. Santa Barbara returned 404 and every other standard disfluency corpus sits behind an LDC agreement.
- 2026-08-04: The SwDA parser emits exactly the `InjectionResult` schema, so `evaluate.py` and `surprisal.py` read real speech with no changes at all.
- 2026-08-04: `{F}` filled pauses, `{E}` editing terms and `{D}` discourse markers are treated as removable; `{C}` coordinating conjunctions and `{A}` asides are kept, since the speaker meant to say them.
- 2026-08-04: Only complete slash units with balanced markup are used, and utterances containing a cut-off word ("d-") are dropped, since nothing in training resembles them. That keeps 81k of 224k utterances.
- 2026-08-04: Added a punctuation-blind WER column. Switchboard transcribers use far more commas than written prose, so the plain WER on real speech partly measures a house-style difference rather than disfluency removal.
- 2026-08-04: Repair resolution had to stop requiring every word of a repair span to survive: real transcripts nest a repair inside the repair half of a larger one, so a perfect output scored 0.88 instead of 1.00.
- 2026-08-04: `data/swda_eval.jsonl` is not committed; it is derived data, regenerable in one command from the URL in the README.
- 2026-08-04: The retrain was interrupted at step 1357; added `--resume auto` and carried on from checkpoint-1200 rather than repeating 30 minutes of work. Loss after the resume (0.2155) matched the values before it (0.2118-0.2233), so the restore was clean; the "missing keys" warning for `embed_tokens` is T5 tying those weights to `shared.weight`.
- 2026-08-04: Deleted the stale `train_summary.json` from the first run as soon as the retrain was interrupted, so a half-finished run could not be reported using the previous run's numbers.
- 2026-08-04: `train_runtime_s` in the summary covers only the resumed leg (2143 s); the full run was about 66 minutes across the interruption.
- 2026-08-04: Repair resolution on the synthetic split fell from 0.945 to 0.848 after the injector was widened. That is the injector getting harder, not the model getting worse.
- 2026-08-04: Growing the test split moved the synthetic repair effect from +7.34 bits (n=64) to +3.84 bits (n=311); the original estimate was inflated by small-sample noise, which is what the larger split was for.
- 2026-08-04: The surprisal legend is placed by matplotlib's "best" rule, since the peak sits in a different corner on synthetic and real data.
- 2026-08-04: Kept both surprisal figures rather than one. The synthetic panel has the tighter sampling and the real panel carries the actual claim, and the difference between them (peak at -1 vs at 0) is itself a finding about the generator.
