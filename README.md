# Mend

Mend turns spoken dictation into written text. It removes filled pauses, resolves
self-repairs, and restores casing and punctuation:

```
send it to john no wait sarah um by friday
-> Send it to Sarah by Friday.
```

There is no corpus of paired dictation and its cleaned-up form, so Mend builds one:
a disfluency injector runs clean written sentences backwards into plausible speech,
keeping a record of exactly what it inserted and where. Those spans are what make
the rest possible — they give the evaluation something to score against, and they
give the surprisal analysis a boundary to centre on.

The repo contains:

* `mend/injector.py` — the disfluency injector, structured the way Shriberg
  describes self-repair: **reparandum** (the abandoned material), **interregnum**
  (the editing phrase), **repair** (what replaces it).
* `mend/swda.py` — a parser for real conversational speech, which emits the same
  schema, so nothing downstream needs to know whether a pair is synthetic or real.
* one fine-tuned model — flan-t5-small, seq2seq, `clean up: {disfluent}` → `{clean}`.
* one evaluation table — `outputs/eval_table.md`.
* the surprisal-at-repair-boundary figures, on synthetic and on real speech.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
```

Any environment with the pinned dependencies in `pyproject.toml` works. The scripts
pick their device automatically: CUDA if present, otherwise Apple MPS, otherwise CPU.

## Running the stages

```bash
# 1. tests
pytest

# 2. synthetic training data: 50k/2k/10k pairs into data/
python -m mend.generate_data --smoke                       # 100 pairs, prints a sample
python -m mend.generate_data --train 50000 --val 2000 --test 10000 --seed 13

# 3. the real-speech evaluation set (14 MB, no licence gate)
curl -L -o swda.zip https://github.com/cgpotts/swda/raw/master/swda.zip
unzip -q swda.zip -d swda
python -m mend.swda --root swda --out data/swda_eval.jsonl

# 4. fine-tune; writes outputs/checkpoints and links outputs/model at the best one
python -m mend.train --smoke --smoke-steps 150 --batch 32  # checks the loss falls
python -m mend.train --batch 32 --epochs 2 --eval-steps 400
#   --resume auto picks up from the latest checkpoint if a run is interrupted

# 5. the evaluation table
python -m mend.evaluate

# 6. the surprisal figures
python -m mend.surprisal --data data/test.jsonl --title "Synthetic test split"
python -m mend.surprisal --data data/swda_eval.jsonl --shuffle --limit 20000 \
    --tag _swda --title "Real speech (Switchboard)"
```

Every stage is seeded and re-runnable. Do not run the surprisal pass next to
training on a 16 GB machine; together they exhaust memory and both crawl.

## Real speech

Everything synthetic in this repo measures one thing: how well the model inverts
this particular injector. To measure anything else you need speech nobody here
designed.

The [Switchboard Dialog Act corpus](https://github.com/cgpotts/swda) is freely
redistributable, is real telephone conversation, and its transcripts carry the
Treebank-3 disfluency annotation — which is the same structure the injector
imitates:

```
[ do you, + do you ] have any experience     repair: reparandum then repair
{F uh, }                                     filled pause
{E I mean, }                                 explicit editing term
{D you know, }                               discourse marker
```

`mend/swda.py` parses that into the same `InjectionResult` schema the injector
produces, so `evaluate.py` and `surprisal.py` read it with no changes. It yields
**81,107 utterances carrying 13,038 repairs, 15,622 repetitions and 35,412 filled
pauses**. Only complete slash units with balanced markup are kept, and utterances
containing a cut-off word ("d-") are dropped since nothing in training resembles
them.

No modelling or data-generation choice in this repo was made by looking at it.

## Results

Trained on 50k pairs for 2 epochs on an M-series Mac, about 66 minutes across an
interrupted-and-resumed run. Validation loss fell at six of seven evaluations,
0.186 → 0.146; the best checkpoint is step 2400.

The copy baseline returns its input with a trivial rule applied — capitalise the
first word, uppercase standalone *i*, add a full stop — which isolates how much of
each score is punctuation restoration alone.

| System | Set | N | Exact match | WER | WER (words only) | Filler removal F1 | Repair resolution | Repairs scored |
|---|---|---|---|---|---|---|---|---|
| Copy baseline | held-out synthetic | 10000 | 0.030 | 0.310 | 0.132 | 0.000 | 0.000 | 2267 |
| Mend (flan-t5-small) | held-out synthetic | 10000 | 0.649 | 0.058 | 0.012 | 0.995 | 0.848 | 2267 |
| Copy baseline | hard set | 100 | 0.080 | 0.689 | 0.601 | 0.000 | 0.000 | 88 |
| Mend (flan-t5-small) | hard set | 100 | 0.550 | 0.206 | 0.171 | 0.667 | 0.557 | 88 |
| Copy baseline | real speech (SwDA) | 2000 | 0.342 | 0.227 | 0.124 | 0.000 | 0.000 | 242 |
| Mend (flan-t5-small) | real speech (SwDA) | 2000 | 0.479 | 0.132 | 0.064 | 0.862 | 0.227 | 242 |

Exact match normalises whitespace and treats the sentence-final mark as optional,
but is otherwise case-sensitive. WER is over whitespace tokens with punctuation
attached, so a missing comma costs one substitution; **WER (words only)** drops
casing and punctuation, which matters for the real-speech row because Switchboard
transcribers comma far more heavily than written prose, and a model trained on
WikiText is marked down there for house style rather than for anything to do with
disfluency. Filler removal F1 counts *occurrences* deleted rather than positions,
because a word like *like* can be a filled pause and a real word in the same
sentence; the scored vocabulary is the injector's filler words plus whatever each
transcript itself annotates as a filler. Repair resolution asks whether the
reparandum is gone and the repair is present, using annotated spans where they
exist and the input/gold word-multiset difference otherwise.

### Surprisal at repair boundaries

Per-token surprisal from pythia-160m, summed within each whitespace word, centred
on the boundary token — the first token of the editing phrase, or of the repair
where there is none. The control is the *same* utterance with its disfluencies
removed, read at the matching word position, so each instance is its own baseline.

| Synthetic | Real speech |
|---|---|
| ![Synthetic](outputs/surprisal_at_repair.png) | ![Real speech](outputs/surprisal_at_repair_swda.png) |

Surprisal spikes at the interruption in both, for all three types, with every
interval excluding zero (95% bootstrap CIs over 4000 resamples of the paired
per-instance differences):

| Type | Synthetic | Real speech (SwDA) |
|---|---|---|
| Repair | +3.84 bits [+2.86, +4.84] at −1 (n=311) | **+4.57 bits [+4.17, +4.98] at 0** (n=536) |
| Repetition | +2.91 bits [+2.23, +3.58] at 0 (n=447) | +5.66 bits [+5.37, +5.95] at 0 (n=728) |
| Filled pause | +5.22 bits [+4.80, +5.65] at +1 (n=1652) | +3.42 bits [+3.10, +3.72] at 0 (n=1430) |

The real-speech result is the one worth quoting: 536 genuine human self-repairs,
the peak landing exactly on the interruption point, an effect of about four and a
half bits.

The two panels differ in a way that indicts the generator. On real speech all three
peaks sit exactly at position 0. On synthetic data the repair peak sits one position
*earlier*, on the reparandum. That is because the injector's reparandum is a word
drawn at random from a category — "send it to *Omar*" when the sentence is about
Sarah — and a random substitution is inherently surprising. A real speaker's
abandoned word is something they were plausibly about to say, so the surprise waits
for the interruption itself. The synthetic figure is measuring the generator's
randomness as much as the phenomenon.

Two further caveats. The reported effects are paired against each type's own
control, while the dashed line pools all control windows, so a gap read straight off
a plot will not reproduce the table. And only disfluencies with a complete ±5 window
on both sides are analysed.

## Honest summary

On held-out synthetic data the model does the job it was trained on: filled pauses
removed essentially perfectly (F1 0.995), 84.8% of injected repairs resolved, WER
cut from 0.310 to 0.058. On real speech, filler removal largely transfers (F1
0.862) and word error rate halves against the baseline (0.124 → 0.064 words-only).

**Repair resolution does not transfer. It falls from 0.848 to 0.227.** That single
number is the most useful result here. Injected repairs are single-category
substitutions with a stock editing phrase; real self-repairs are none of those
things, and a model that has only ever seen the synthetic kind resolves fewer than
a quarter of them. Anyone reading the 0.848 as evidence of a dictation cleanup
system would be wrong by a factor of nearly four.

The characteristic real-speech failures are:

* **Over-deletion of literal fillers.** *i like to eat italian food best* comes back
  as *Eat Italian food best.* — *like* was treated as a filled pause even after
  30% of training targets were made to contain a literal filler-like word.
* **Disagreement about discourse markers.** Switchboard marks sentence-initial
  *well* as removable; the model keeps it. This is a genuine annotation-convention
  difference, not straightforwardly a model error.
* **Occasional hallucinated continuations.** *if you make homemade bread* →
  *If you make homemade bread, you can.*

### What the injector changes bought

The injector was widened after the first round of hard-set failures: repairs now
span up to three tokens and overlap their reparandum, corrections can arrive late
(the cross-clause case), false starts can happen mid-sentence, and filler words
appear literally in a third of targets. Hard-set exact match by phenomenon, before
and after:

| Phenomenon | Before | After |
|---|---|---|
| Overlapping repair | 8/14 | **11/14** |
| Cross-clause repair | 1/12 | **4/12** |
| False start | 1/8 | 2/8 |
| Filler used literally | 7/14 | 8/14 |
| Repetition | 7/8 | 7/8 |
| Number and date repair | 13/16 | 12/16 |
| Multi-repair | 12/16 | **9/16** |
| Nested repair | 4/12 | **2/12** |

The two categories the changes targeted improved; multi-repair and nested repair
got worse, and overall hard-set exact match barely moved (0.530 → 0.550). Widening
the injector traded one failure mode for another rather than lifting the ceiling.
**With 8–16 examples per phenomenon these per-row differences are not statistically
meaningful** — read them as directions, not measurements.

### On the status of the hard set

The hard set was built to probe failure modes, and then the training distribution
was changed in response to what it found. That makes it a **development set**, not
a held-out measurement, for the filler-as-content and repair-shape categories. Its
numbers are optimistic to an unknown degree. SwDA is the honest benchmark: it was
parsed after every modelling decision had been made, and nothing was tuned against
it.

### Limitations

All training data is synthetic — no real ASR output and no real disfluent speech
goes into the model, only into its evaluation. The injector's repairs are drawn
from closed word categories. WikiText prose and dictation templates are the only
two registers. SwDA is telephone conversation, which is not the dictation register
the model targets, so part of the real-speech gap is domain rather than disfluency.
And the surprisal analysis, though it rests on 536 real repairs, uses a 160M-parameter
LM whose absolute bit values would shift with a larger one — the direction and the
position of the peak are the robust parts, not the magnitude.

## A note on `data/hard_set.jsonl`

The hard set is 100 adversarial examples covering nested repairs, repairs that
cross a clause boundary, repairs whose reparandum and repair share words, words
like *like* and *sorry* used literally rather than as fillers, number and date
repairs, and sentences carrying several repairs at once. It was drafted in one
pass and **has not been reviewed by a second pair of eyes**. Read it and edit it
by hand before quoting any hard-set number as a result — it is the source of every
number in the middle of the results table, and some of its gold answers are
debatable (whether *the flight leaves at 6 45 no wait 7 45* should require the
model to produce `7:45`, for one).

## Layout

```
mend/injector.py       disfluency injection with reparandum/interregnum/repair spans
mend/generate_data.py  WikiText + dictation templates -> (disfluent, clean) pairs
mend/swda.py           Switchboard transcripts -> the same schema, from real speech
mend/train.py          seq2seq fine-tune
mend/evaluate.py       the results table
mend/surprisal.py      per-token surprisal and the figures
data/hard_set.jsonl    100 hand-written hard cases
outputs/               table, figures, predictions, checkpoints
DECISIONS.md           one line per engineering decision made along the way
```
