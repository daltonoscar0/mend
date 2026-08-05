# Mend

[![tests](https://github.com/daltonoscar0/mend/actions/workflows/tests.yml/badge.svg)](https://github.com/daltonoscar0/mend/actions/workflows/tests.yml)

Mend turns spoken dictation into written text: it removes filled pauses, resolves
self-repairs, and restores casing and punctuation.

```
send it to john no wait sarah um by friday
-> Send it to Sarah by Friday.
```

There is no corpus of paired dictation and its cleaned-up form, so Mend builds
one. A disfluency injector runs clean written sentences backwards into plausible
speech, keeping a record of exactly what it inserted and where — and those spans
are what make the rest possible, because they give the evaluation something to
score against.

## The finding

A model trained only on synthetically injected disfluencies resolves **about a
third** of real human self-repairs. Adding real speech to the training mix takes
it to **about a half**.

| | repair resolution on real speech |
|---|---|
| copy baseline | 0.000 |
| trained on synthetic disfluencies only | 0.326 [0.285, 0.366] |
| **+ 25k real utterances** | **0.503 [0.460, 0.546]** |

Measured on 293 Switchboard conversations neither model was trained on. Those two
intervals do not overlap, and that is the one comparison in this repo with a
comfortable margin.

Almost nothing else does. The mixed model also looks better on the synthetic
split and on the hand-written hard set, and in both cases the intervals overlap —
so the honest reading is "did not regress", not "improved". Two further claims
that were in this README have since been withdrawn: a hard-set result that 100
sentences could not support, and an argument about the generator that did not
survive being re-run with a larger language model. Both retractions are below,
with the reasoning left intact.

Half of real self-repairs are still unresolved. That is the honest ceiling.

## What is here

* `mend/injector.py` — the disfluency injector, structured the way Shriberg
  describes self-repair: **reparandum** (the abandoned material), **interregnum**
  (the editing phrase), **repair** (what replaces it).
* `mend/swda.py` — a parser for real conversational speech, which emits the same
  schema, so nothing downstream needs to know whether a pair is synthetic or real.
* two fine-tuned models — flan-t5-small, seq2seq, `clean up: {disfluent}` →
  `{clean}` — one trained on synthetic pairs alone and one with real speech mixed
  in, so the difference between them is measurable rather than asserted.
* one evaluation table, `outputs/eval_table.md`, and the
  surprisal-at-repair-boundary figures.
* `DECISIONS.md` — one line per engineering decision, including the ones that
  turned out to be wrong.

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
pauses**. Only complete slash units with balanced markup are kept, and a slash
unit that breaks off mid-word is dropped.

Cut-off words *inside* an utterance are not. An earlier version of this file
claimed they were filtered out, and they never were: `spoken_form` strips the
trailing hyphen while building the tokens, so by the time the filter ran there
was nothing left to match and it fired on 0 of the 81,107 utterances. "ma-"
enters the corpus as "ma", indistinguishable from a word the speaker meant. The
dead branch has been removed and this paragraph corrected rather than the corpus
rebuilt, since the shipped models were trained on it.

### The split, and what it costs

Real speech now feeds training as well as evaluation, because the first round
showed that synthetic repairs alone do not teach the model to resolve real ones.
So the corpus is divided **by conversation**, never by utterance:

| Split | Conversations | Utterances | Repairs |
|---|---|---|---|
| train | 804 | 56,684 | 9,199 |
| val | 58 | 4,109 | 583 |
| eval | 293 | 20,314 | 3,256 |

Splitting by utterance would leak. The two speakers in a Switchboard call talk
about one assigned topic for ten minutes, so utterances from the same call share
vocabulary, subject matter and speaker habits; a model trained on half a call and
tested on the other half is being asked an easier question than the one we mean
to ask. Whole conversations therefore go to exactly one split.

This does spend something. In the first round nothing in the repo had ever been
fitted to SwDA, which made every number on it a clean zero-shot transfer result.
That is no longer true of the corpus as a whole — but the 293 evaluation
conversations have still never been trained on, and the synthetic-only model is
kept and re-scored on that same held-out set, so the before/after comparison
below is like for like. The zero-shot transfer finding is preserved as its own
row rather than overwritten.

The surprisal analysis reads the *whole* parse, not the held-out split: it
measures a property of human speech using a separate language model, so which
conversations Mend trained on has no bearing on it.

## Results

Two systems, identical except for what they were trained on:

* **Mend (synthetic only)** — 50k injected pairs, 2 epochs, about 66 minutes on an
  M-series Mac across an interrupted-and-resumed run. Validation loss 0.186 →
  0.146; best checkpoint step 2400.
* **Mend (+ real speech)** — the same 50k plus 25k real Switchboard utterances
  (33% of the mix), 2 epochs, 62 minutes. Validation loss 0.216 → 0.159; best
  checkpoint step 4000. The two validation figures are not comparable to each
  other: the second validation set contains real speech and is harder.

The copy baseline returns its input with a trivial rule applied — capitalise the
first word, uppercase standalone *i*, add a full stop — which isolates how much of
each score is punctuation restoration alone.

| System | Set | N | Exact match | WER | WER (words only) | Filler removal F1 | Repair resolution | Repairs scored |
|---|---|---|---|---|---|---|---|---|
| Copy baseline | held-out synthetic | 10000 | 0.030 [0.027, 0.033] | 0.310 | 0.132 | 0.000 | 0.000 [0.000, 0.000] | 2267 |
| Mend (synthetic only) | held-out synthetic | 10000 | 0.649 [0.639, 0.658] | 0.058 | 0.012 | 0.995 | 0.848 [0.832, 0.861] | 2267 |
| Mend (+ real speech) | held-out synthetic | 10000 | 0.646 [0.637, 0.655] | 0.058 | 0.012 | 0.996 | 0.863 [0.848, 0.877] | 2267 |
| Copy baseline | hard set | 100 | 0.080 [0.030, 0.140] | 0.684 | 0.597 | 0.000 | 0.000 [0.000, 0.000] | 88 |
| Mend (synthetic only) | hard set | 100 | 0.550 [0.450, 0.650] | 0.204 | 0.168 | 0.667 | 0.568 [0.460, 0.670] | 88 |
| Mend (+ real speech) | hard set | 100 | 0.610 [0.520, 0.710] | 0.177 | 0.143 | 0.571 | 0.648 [0.548, 0.747] | 88 |
| Copy baseline | real speech (SwDA) | 4000 | 0.325 [0.310, 0.339] | 0.222 | 0.117 | 0.000 | 0.000 [0.000, 0.000] | 519 |
| Mend (synthetic only) | real speech (SwDA) | 4000 | 0.455 [0.439, 0.470] | 0.125 | 0.056 | 0.859 | 0.326 [0.285, 0.366] | 519 |
| Mend (+ real speech) | real speech (SwDA) | 4000 | **0.616** [0.600, 0.631] | 0.073 | **0.026** | **0.955** | **0.503** [0.460, 0.546] | 519 |

The real-speech rows are the 293 held-out conversations, which neither model was
trained on. Bracketed ranges are 95% percentile bootstrap intervals over 2000
resamples, resampling **sentences** — a sentence is what was independently drawn
from the corpus, and the several disfluencies inside one are not independent of
each other, so resampling repairs instead would give intervals that are too
narrow.

### What that difference looks like

Real Switchboard utterances from the held-out conversations, with both models'
output. These are the disfluency cases rather than the punctuation ones — the
speaker abandons something and starts again, and the question is whether the
abandoned material survives into the written form:

```
in    they had they've scaled everything down so much
gold  They've scaled everything down so much.
syn   They had they scaled everything down so much.        <- keeps both attempts
mix   They've scaled everything down so much.

in    i think i heard that was just one was just out
gold  I think I heard that was just out.
syn   I think I heard that was just one was just out.      <- keeps both attempts
mix   I think I heard that was just out.

in    so it work it did end up working out pretty well
gold  So it did end up working out pretty well?
syn   So it work, it did end up working out pretty well.   <- punctuates the seam
mix   So it did end up working out pretty well.

in    are you going to go to east texas for that
gold  Are you going to go to East Texas for that?
syn   Go to Texas for that.                                <- deletes real words
mix   Are you going to go to East Texas for that?
```

The synthetic-only model has two characteristic failures here, and they pull in
opposite directions. It leaves an abandoned phrase in place when the repair does
not look like the injector's stock `no wait X` shape, and it deletes genuine
words when a sentence opens the way a disfluency does. The last case is the
worst kind of error a cleanup tool can make: it did not fail to tidy the input,
it threw away half of what the speaker said.

It still gets plenty wrong:

```
in    and then i it's like well the thrill is gone
gold  And then it's like the thrill is gone.
mix   And then I, it's like, the thrill is gone.           <- abandoned "i" kept

in    degree it's of course it's a whole lot easier
gold  Degree, of course, it's a whole lot easier.
mix   Degree, it's a whole lot easier.                     <- drops "of course"
```

Half of real self-repairs still go unresolved, and these are what that looks
like.

Only the real-speech row is bolded, because it is the only one where the two
models are actually separated. **On real speech the intervals do not come close
to overlapping** (0.326 [0.285, 0.366] against 0.503 [0.460, 0.546]). Everywhere
else they overlap: the synthetic rows say the mixed model did not regress, not
that it improved, and the hard set at 100 sentences cannot separate 0.550 from
0.610 — that row is a direction, not a measurement.

> **The earlier version of this table reported 0.227 for repair resolution on real
> speech, not 0.326.** That was the same checkpoint measured on a different sample:
> 2000 utterances drawn from the whole corpus, carrying 242 scored repairs, before
> any split existed. Re-running the synthetic-only model under that old sampling
> now gives 0.260 on 265 repairs — the spread between 0.227 and 0.260 is what a
> couple of hundred scored repairs buys you. The held-out sample sits a little
> higher still at 0.326, which is what you would expect if conversations vary in
> difficulty, though at these sample sizes the intervals overlap and the gap is
> not firmly established. Nothing about the model changed; the measurement did.
> Both systems above are scored on the identical sample, so
> **0.326 → 0.503 is the like-for-like comparison**, and the evaluation sample was
> doubled to 4000 utterances so the headline number rests on 519 sentences
> carrying 637 repairs. (The metric is per sentence: a sentence with two repairs
> scores only if both came out right.)

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

The same two panels from pythia-1.4b are in
[`outputs/surprisal_at_repair_p14b.png`](outputs/surprisal_at_repair_p14b.png)
and
[`outputs/surprisal_at_repair_swda_p14b.png`](outputs/surprisal_at_repair_swda_p14b.png);
the comparison is below and it matters for the synthetic panel.

Surprisal spikes at the interruption in both, for all three types, with every
interval excluding zero (95% bootstrap CIs over 4000 resamples of the paired
per-instance differences):

| Type | Synthetic | Real speech (SwDA) |
|---|---|---|
| Repair | +3.84 bits [+2.86, +4.84] at −1 (n=311) | **+4.82 bits [+4.42, +5.21] at 0** (n=548) |
| Repetition | +2.91 bits [+2.23, +3.58] at 0 (n=447) | +5.75 bits [+5.45, +6.05] at 0 (n=653) |
| Filled pause | +5.22 bits [+4.80, +5.65] at +1 (n=1652) | +3.09 bits [+2.76, +3.43] at 0 (n=1430) |

The real-speech result is the one worth quoting: 548 genuine human self-repairs,
the peak landing exactly on the interruption point, an effect of about five bits.

This analysis reads the whole corpus, both the conversations Mend trained on and
the ones it did not, because it measures human speech with a separate language
model and Mend plays no part in it. An earlier run over a different 20,000-utterance
sample of the same corpus gave +4.57 [+4.17, +4.98] on 536 repairs, +5.66 on
repetitions and +3.42 on filled pauses — the same three peaks in the same places,
which is the reassuring part.

### Does this survive a bigger language model?

The numbers above come from pythia-160m, and an obvious objection is that they
measure a small model's limitations rather than a property of speech. Re-running
the identical samples through **pythia-1.4b**, nine times the parameters, same
family:

| Type | Real speech, 160m | Real speech, 1.4b | Synthetic, 160m | Synthetic, 1.4b |
|---|---|---|---|---|
| Repair | +4.82 at 0 | +4.93 at 0 | +3.84 at **−1** | +5.06 at **0** |
| Repetition | +5.75 at 0 | +5.83 at 0 | +2.91 at 0 | +5.26 at 0 |
| Filled pause | +3.09 at 0 | +2.43 at 0 | +5.22 at **+1** | +5.81 at **0** |

**On real speech the result is solid.** Every peak stays exactly at the
interruption point and the two largest effects barely move — repair +4.82 →
+4.93, repetition +5.75 → +5.83, well inside each other's intervals. Only the
filled-pause magnitude shifts down materially (+3.09 → +2.43). The headline
claim — that a language model's surprise spikes at the interruption, worth about
five bits, over 548 genuine human self-repairs — does not depend on model size.

**On synthetic data it is not solid, and this cost me a conclusion.** An earlier
version of this README argued that the two panels differed in a way that
indicted the generator: on real speech all three peaks sat at 0, while on
synthetic data the repair peak sat one position *earlier*, on the reparandum.
The explanation offered was that the injector's reparandum is a word drawn at
random from a category — "send it to *Omar*" when the sentence is about Sarah —
and a random substitution is inherently surprising, whereas a real speaker's
abandoned word is something they were plausibly about to say.

pythia-1.4b puts all three synthetic peaks at 0, exactly like real speech. The
panels no longer differ, so the offset was an artefact of the smaller model
rather than a fingerprint of the generator. Looking back at the 160m numbers the
signal was always thin: the synthetic repair effect was +3.84 at −1 against
+3.26 at 0, two nearly equal values with overlapping intervals, and `argmax`
had to pick one.

The criticism of the injector still stands on its own terms — its reparandum
usually *is* a random category substitution — 88% of injected repairs are, and
69% carry one of four stock editing phrases — which is a fact about the code
rather than something this analysis discovered. What is gone is the evidence that the
surprisal figure demonstrated it.

Two further caveats. The reported effects are paired against each type's own
control, while the dashed line pools all control windows, so a gap read straight off
a plot will not reproduce the table. And only disfluencies with a complete ±5 window
on both sides are analysed.

## Honest summary

The single most misleading number in this repo is **0.848** — the synthetic-only
model's repair resolution on held-out synthetic data. The same model scores
**0.326** on real speech. Anyone reading the first as evidence of a dictation
cleanup system would be wrong by roughly a factor of three.

That gap is the whole point. Injected repairs are mostly category substitutions
(88%), two thirds of them announced by one of four stock editing phrases; real
self-repairs are far less uniform. Whether a differently shaped generator could
have closed more of the gap was never tested — the injector was widened once and
the pre-widening model was never scored on real speech.

Adding 25k real utterances takes real-speech repair resolution to
**0.503 [0.460, 0.546]**, cuts words-only WER from 0.056 to **0.026**, and lifts
filler removal from 0.859 to **0.955**. Those intervals are far apart; this is
the one comparison here with a comfortable margin.

What it does *not* do is improve anything else measurably. On the synthetic split
the intervals overlap in both directions (exact match 0.649 → 0.646, repair
resolution 0.848 → 0.863), which supports "did not regress" and nothing stronger.
The hard set moves 0.550 → 0.610, but at 100 sentences that interval is ±0.10 and
the two models are not separated. The useful reading is that the worry — that
telephone conversation would drag the model off the dictation register — did not
materialise at a third of the mix, rather than that mixing helped everywhere.

Half of real self-repairs still go unresolved. That remains the honest ceiling.

### How much of the real-speech gain is really disfluency?

Some of it is house style. The mixed model has learned Switchboard's comma
conventions along with its disfluencies — *so uh what was the last car repair you
had* comes back as *So, what was...* rather than *So what was...*, which is worth
a chunk of the plain-WER improvement and nothing at all in cleanup quality. Three
things argue the gain is not only that:

* **words-only WER**, which is blind to punctuation and casing, still more than
  halves (0.056 → 0.026);
* **repair resolution** is a word-multiset test that ignores punctuation entirely;
* the failures that disappear are real ones. *are you going to go to east texas
  for that* came back from the synthetic-only model as *Go to Texas for that.* —
  a catastrophic over-deletion — and from the mixed model correctly.

The caveat that remains: the model is now trained on SwDA's annotation
conventions and evaluated against them. That is legitimate for measuring whether
it learned real disfluency structure, and it would flatter the model against any
corpus that draws the removable/keepable line somewhere else.

### The characteristic failures, after mixing

* **Over-deletion of literal fillers is largely fixed but traded.** Filler removal
  on real speech rose to 0.955, but hard-set *filler-as-content* fell 8/14 → 7/14.
  The model is better calibrated on conversational fillers and slightly worse on
  the adversarial dictation cases.
* **Fragments are still unhandled**, and not for the reason this README used to
  give. *it sure does ma make a difference* keeps the cut-off *ma*. Training does
  contain such utterances — the filter meant to remove them never fired — but the
  hyphen is stripped on the way in, so *ma-* arrives as *ma* with nothing marking
  it as a fragment, and the model has no signal to learn from.
* **Discourse-marker disagreement is now mostly resolved** in Switchboard's
  favour, which is a convention the model has been taught rather than a judgement
  it makes.

### Hard set by phenomenon

Exact match on each of the eight adversarial categories, both models:

| Phenomenon | synthetic only | + real speech |
|---|---|---|
| Overlapping repair | 11/14 | **12/14** |
| Cross-clause repair | 4/12 | **5/12** |
| False start | 2/8 | 2/8 |
| Filler used literally | 8/14 | 7/14 |
| Repetition | 7/8 | 7/8 |
| Number and date repair | 12/16 | **14/16** |
| Multi-repair | 9/16 | **10/16** |
| Nested repair | 2/12 | **4/12** |
| **total** | **55/100** | **61/100** |

Seven of eight categories improve or hold — five improve, two hold — and the
one that regresses is
filler-as-content — the model traded a little of its dictation-register
calibration for a lot of conversational-filler accuracy. **With 8–16 examples per
phenomenon these per-row differences are not statistically meaningful** — read
them as directions, not measurements. Nested repair doubled to 4/12, which
leaves it and false start (flat at 2/8) as the two weakest categories; neither
training mix has touched false starts at all.

An earlier round widened the injector itself — three-token repairs that overlap
their reparandum, corrections that arrive a clause late, mid-sentence false
starts, literal filler words in a third of targets. That moved hard-set exact
match 0.530 → 0.550, trading multi-repair and nested repair for overlapping and
cross-clause. The hard set cannot separate either move at n=100 — widening the
generator visibly rearranged the failures, and the gain that is firmly measured
anywhere is on real speech.

### On the status of the hard set

The hard set was built to probe failure modes, and then the training distribution
was changed in response to what it found. That makes it a **development set**, not
a held-out measurement, for the filler-as-content and repair-shape categories. Its
numbers are optimistic to an unknown degree.

The 293 held-out SwDA conversations are the honest benchmark. They were never
trained on, and the split is a seeded shuffle over whole conversations drawn
before the mixed model was trained; the synthetic-only model predates the split
entirely and had never seen SwDA at all. The
corpus as a whole no longer has the stronger property it had in the previous
round — that nothing in the repo had ever been fitted to any part of it.

### Limitations

Real disfluent speech now reaches the model, but only Switchboard telephone
conversation, and only through a transcriber's annotation conventions; the model
has been taught where *that* corpus draws the removable line. There is still no
real ASR output anywhere in the pipeline — the disfluent side is always a
transcript, never a recognition hypothesis, so it carries none of the substitution
errors a real recogniser makes. The injector's repairs are still drawn from closed
word categories. Cut-off words reach training with their hyphen stripped, so
nothing distinguishes a fragment from a real word and fragments are unhandled. The surprisal analysis rests
on 548 real repairs and its real-speech peaks hold at nine times the model size,
but the synthetic peaks do not, so any claim resting on where a synthetic peak
falls should be treated as a property of the language model until it has been
checked at more than one scale — as one such claim in this README turned out to
be. Absolute bit values shift with model size in both.

## A note on `data/hard_set.jsonl`

The hard set is 100 adversarial examples covering nested repairs, repairs that
cross a clause boundary, repairs whose reparandum and repair share words, words
like *like* and *sorry* used literally rather than as fillers, number and date
repairs, and sentences carrying several repairs at once.

It has since been reviewed. Every gold was checked to introduce no word absent
from its input, apostrophes aside — no example asks the model to invent a word
the speaker never said — and there are no duplicate inputs or golds. The twelve
cross-clause rows are not reachable by deletion alone, though: the corrected
phrase has to move back to where the abandoned one stood. One
inconsistency was found and fixed: *the flight leaves at 6 45 no wait 7 45* asked
for `7:45`, while every other number row copies its digits verbatim (`555 0128`
keeps its space, *at 3 no at 4* stays a bare `4`). The set's convention is that
repairs get resolved, not that numbers get reformatted, so the gold is now `7 45`.

Two things to know before quoting a hard-set number. Exact match on four rows
turns on orthography rather than disfluency — *like-minded*, *north-east*,
*double-sided*, and the apostrophe in *Let's* — which is legitimate written
English but is not what the set was built to probe. And it remains a development
set for the categories the injector was tuned against.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
```

Any environment with the pinned dependencies in `pyproject.toml` works. The scripts
pick their device automatically: CUDA if present, otherwise Apple MPS, otherwise CPU.

## Using it

Once there is a model at `outputs/model`:

```console
$ mend-clean "um send the the report to john no wait sarah by friday"
Send the report to Sarah by Friday.

$ echo "so i was like really tired you know" | mend-clean
So I was really tired.
```

It reads stdin when given no arguments, one line in, one line out, so it works in
a pipe. Input is normalised to the register the model was trained on — lowercased,
edge punctuation removed — so pasting text that already has capitals and commas
works; `--raw` skips that. `--show-input` prints both sides.

`outputs/model` points at the model trained with real speech mixed in: clearly
better on real speech, and no worse on the synthetic and hard sets. Read the
accuracy section before trusting it on anything — it removes filled pauses and
restores punctuation reliably, and resolves about half of real self-repairs.

## Running the stages

```bash
# 1. tests: no checkpoint and no corpus download needed
pytest

# 2. synthetic training data: 50k/2k/10k pairs into data/
python -m mend.generate_data --smoke                       # 100 pairs, prints a sample
python -m mend.generate_data --train 50000 --val 2000 --test 10000 --seed 13

# 3. real speech (14 MB, no licence gate). Writes the whole parse for the
#    surprisal analysis, plus a conversation-disjoint train/val/eval beside it
curl -L -o swda.zip https://github.com/cgpotts/swda/raw/master/swda.zip
unzip -q swda.zip -d swda
python -m mend.swda --root swda --out data/swda_all.jsonl --split

# 4. fine-tune; writes <out>/checkpoints and links <out>/model at the best one
python -m mend.train --smoke --smoke-steps 150 --batch 32 --real 25000
#   the two systems in the table: synthetic only, then with real speech mixed in
python -m mend.train --batch 32 --epochs 2 --eval-steps 400 \
    --real 0 --out outputs/synthetic_only
python -m mend.train --batch 32 --epochs 2 --eval-steps 400 \
    --real 25000 --out outputs/mixed
#   --real -1 uses every real pair instead of a 25k sample
#   --resume auto picks up from the latest checkpoint if a run is interrupted
ln -sfn mixed/model outputs/model   # what mend-clean uses; train.py already
#                                    keeps <out>/model on the best checkpoint

# 5. the evaluation table; --model takes "label=path" and repeats
python -m mend.evaluate --bootstrap 2000 \
    --model "Mend (synthetic only)=outputs/synthetic_only/model" \
    --model "Mend (+ real speech)=outputs/mixed/model"

# 6. the surprisal figures
python -m mend.surprisal --data data/test.jsonl --draws 4000 \
    --title "Synthetic test split"
python -m mend.surprisal --data data/swda_all.jsonl --shuffle --limit 20000 \
    --draws 4000 --tag _swda --title "Real speech (Switchboard)"
#   the same two at 9x the model size, which is what --lm is for
python -m mend.surprisal --data data/test.jsonl --draws 4000 \
    --lm EleutherAI/pythia-1.4b --tag _p14b --title "Synthetic, pythia-1.4b"
python -m mend.surprisal --data data/swda_all.jsonl --shuffle --limit 20000 \
    --draws 4000 --lm EleutherAI/pythia-1.4b --tag _swda_p14b \
    --title "Real speech, pythia-1.4b"

# 7. run the model over your own text
mend-clean "um send the the report to john no wait sarah by friday"
```

Every stage is seeded and re-runnable. Do not run the surprisal pass next to
training on a 16 GB machine; together they exhaust memory and both crawl.

The test suite builds its own data and stubs the language model, so it runs on a
fresh clone in seconds and needs no GPU — which is what lets CI run it on every
push. It covers the injector, the SwDA parser and split, the
training mix, the scoring functions and the surprisal windowing. `train.py`'s
loop and `generate_data.py`'s WikiText reader are not covered; both are thin
wrappers over other people's libraries and neither produces a reported number.

## Layout

```
mend/injector.py       disfluency injection with reparandum/interregnum/repair spans
mend/generate_data.py  WikiText + dictation templates -> (disfluent, clean) pairs
mend/swda.py           Switchboard transcripts -> the same schema, from real speech,
                       and the conversation-disjoint split
mend/train.py          seq2seq fine-tune, synthetic pairs with real speech mixed in
mend/evaluate.py       the results table, one row per system per set
mend/surprisal.py      per-token surprisal and the figures
mend/clean.py          the mend-clean command, for running the model over text
data/hard_set.jsonl    100 hand-written hard cases
outputs/synthetic_only/model  trained on injected pairs alone
outputs/mixed/model           trained with 25k real utterances mixed in
outputs/model          symlink to whichever of the two mend-clean should use
outputs/               table, figures, predictions, checkpoints
DECISIONS.md           one line per engineering decision made along the way
```
