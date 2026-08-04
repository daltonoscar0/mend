"""Evaluate the cleanup model and write the results table.

Two evaluation sets:

* the held-out synthetic test split, which carries injector annotations, so
  filler deletion and repair resolution can be scored against known spans;
* the hand-written hard set, which has no annotations, so the same two metrics
  are derived from the word-multiset difference between input and gold.

Two systems: the fine-tuned model, and a copy baseline that returns its input
with a trivial casing and punctuation rule applied, which shows how much of the
score comes from punctuation restoration alone.

Usage::

    python -m mend.evaluate
    python -m mend.evaluate --limit 200          # quick pass
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from mend.generate_data import read_jsonl
from mend.injector import FILLERS

# Every word the injector can use as a filled pause. Real speech uses a wider
# range ("oh", "well", "so"), so the scored vocabulary is this set plus
# whatever a given row actually annotates as a filler.
FILLER_VOCAB = frozenset(w for phrase in FILLERS for w in phrase)

_PUNCT = re.compile(r"[^\w\s']|_")


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def match_form(text: str) -> str:
    """Whitespace-normalised, with the sentence-final mark treated as optional."""
    return normalize_ws(text).rstrip(".!?")


def word_tokens(text: str) -> List[str]:
    """Lowercased, punctuation-stripped tokens, for the count-based metrics."""
    out = []
    for tok in normalize_ws(text).lower().split():
        tok = _PUNCT.sub("", tok)
        if tok:
            out.append(tok)
    return out


def count_form(token: str) -> Optional[str]:
    """A single annotation token in the same shape the counters use.

    Annotation tokens keep internal hyphens ("uh-huh") while the counters strip
    them, so the two have to be put through the same normaliser before their
    keys can be compared.
    """
    parts = word_tokens(token)
    return parts[0] if parts else None


# --------------------------------------------------------------------------
# the copy baseline
# --------------------------------------------------------------------------


def copy_baseline(disfluent: str) -> str:
    """Return the input with a trivial casing and punctuation rule applied."""
    text = normalize_ws(disfluent)
    if not text:
        return text
    words = [("I" if w == "i" else w) for w in text.split()]
    text = " ".join(words)
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def exact_match(hyp: str, gold: str) -> bool:
    return match_form(hyp) == match_form(gold)


def word_error_rate(hyps: Sequence[str], golds: Sequence[str]) -> float:
    """Corpus WER over whitespace tokens, punctuation included.

    Punctuation travels attached to its word, so a missing comma counts as one
    substitution. That is deliberate: restoring punctuation is part of the job.
    """
    import jiwer

    refs = [normalize_ws(g) for g in golds]
    outs = [normalize_ws(h) if h.strip() else "@" for h in hyps]
    return float(jiwer.wer(refs, outs))


def word_error_rate_spoken(hyps: Sequence[str], golds: Sequence[str]) -> float:
    """WER over lowercased, punctuation-free tokens.

    Needed to read the real-speech row honestly. Switchboard transcribers comma
    far more heavily than written prose does, so a model trained on WikiText
    punctuation is marked down there for a house-style difference rather than
    for anything to do with disfluency. This variant asks only whether the
    right words came out in the right order.
    """
    import jiwer

    refs = [" ".join(word_tokens(g)) or "@" for g in golds]
    outs = [" ".join(word_tokens(h)) or "@" for h in hyps]
    return float(jiwer.wer(refs, outs))


def filler_vocab_for(row: dict) -> frozenset:
    """Filler words to score in this row.

    The global set alone would miss the fillers real transcripts use, and the
    row's own annotations alone would miss the case that matters most: a word
    deleted as a filler when this row used it literally. Both are needed.
    """
    tokens = row.get("disfluent_tokens")
    anns = row.get("annotations")
    if not tokens or not anns:
        return FILLER_VOCAB
    extra = set()
    for ann in anns:
        if ann["type"] != "filler":
            continue
        span = ann.get("interregnum")
        if span is None:
            continue
        for i in range(span["start"], span["end"]):
            word = count_form(tokens[i])
            if word:
                extra.add(word)
    return frozenset(FILLER_VOCAB | extra)


def gold_filler_deletions(row: dict) -> Tuple[Counter, Counter]:
    """Filler-vocabulary tokens cleanup should remove, split by why.

    Returns (deletions owed to filled pauses, deletions owed to other
    disfluency types). The second one matters because a false start like
    "can you--" also deletes *you*, which belongs to the filler phrase
    "you know"; charging that deletion against filler precision would penalise
    a perfect output. Words appearing there are left out of the score for that
    row.

    Taken from the injector annotations when they exist. The hard set has none,
    so the quantity is recovered from the input/gold difference restricted to
    the filler vocabulary; on the synthetic split the two agree exactly.
    """
    tokens = row.get("disfluent_tokens")
    anns = row.get("annotations")
    if not tokens or anns is None:
        diff = Counter(word_tokens(row["disfluent"])) - Counter(word_tokens(row["clean"]))
        return Counter({w: c for w, c in diff.items() if w in FILLER_VOCAB}), Counter()

    vocab = filler_vocab_for(row)
    fillers: Counter = Counter()
    others: Counter = Counter()
    for ann in anns:
        bucket = fillers if ann["type"] == "filler" else others
        for key in ("reparandum", "interregnum"):
            span = ann.get(key)
            if span is None:
                continue
            for i in range(span["start"], span["end"]):
                word = count_form(tokens[i])
                if word and word in vocab:
                    bucket[word] += 1
    return fillers, others


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "PRF") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def filler_removal_counts(row: dict, hyp: str) -> PRF:
    """Score filler deletion by counting occurrences, not positions.

    A word like *like* can be both a filled pause and a real word in the same
    sentence, so the question is how many copies of it survived, not whether
    any did.
    """
    gold_del, other_del = gold_filler_deletions(row)
    vocab = filler_vocab_for(row)
    src = Counter(word_tokens(row["disfluent"]))
    out = Counter(word_tokens(hyp))
    prf = PRF()
    for word in (set(gold_del) | (vocab & set(src))) - set(other_del):
        wanted = gold_del.get(word, 0)
        removed = max(0, src.get(word, 0) - out.get(word, 0))
        prf.tp += min(wanted, removed)
        prf.fp += max(0, removed - wanted)  # deleted a legitimate occurrence
        prf.fn += max(0, wanted - removed)  # left a filler in
    return prf


def repair_cases_annotated(row: dict) -> List[Tuple[str, str]]:
    """(reparandum, repair) word pairs from the injector annotations."""
    tokens = row["disfluent_tokens"]
    cases = []
    for ann in row["annotations"]:
        if ann["type"] != "repair":
            continue
        rep = ann["reparandum"]
        fix = ann["repair"]
        cases.append(
            (
                " ".join(tokens[rep["start"] : rep["end"]]),
                " ".join(tokens[fix["start"] : fix["end"]]),
            )
        )
    return cases


def repair_resolved_annotated(row: dict, hyp: str) -> Optional[bool]:
    """True when every annotated repair in the row came out right.

    A repair counts as resolved when the reparandum is gone from the output
    (allowing for the times that word legitimately appears in the target too)
    and the repair word is present.
    """
    cases = repair_cases_annotated(row)
    if not cases:
        return None
    out = Counter(word_tokens(hyp))
    gold = Counter(word_tokens(row["clean"]))
    for reparandum, repair in cases:
        for word in word_tokens(reparandum):
            if out.get(word, 0) > gold.get(word, 0):
                return False
        for word in word_tokens(repair):
            # A repair span can enclose material that some other annotation
            # deletes -- real transcripts nest repairs inside the repair half
            # of a bigger one. Only words that survive into the gold are
            # required to survive into the output.
            if gold.get(word, 0) and out.get(word, 0) < 1:
                return False
    return True


def repair_resolved_unannotated(row: dict, hyp: str) -> Optional[bool]:
    """Same question for rows with no spans, from the input/gold difference.

    Material present in the input but not in the gold is what the speaker
    abandoned; material in the gold is what has to survive.
    """
    src = Counter(word_tokens(row["disfluent"]))
    gold = Counter(word_tokens(row["clean"]))
    abandoned = src - gold
    if not abandoned:
        return None
    out = Counter(word_tokens(hyp))
    if any((out - gold).get(word, 0) > 0 for word in abandoned):
        return False  # abandoned material survived
    if gold - out:
        return False  # the corrected material did not
    return True


def repair_resolved(row: dict, hyp: str) -> Optional[bool]:
    if row.get("annotations"):
        return repair_resolved_annotated(row, hyp)
    return repair_resolved_unannotated(row, hyp)


# --------------------------------------------------------------------------
# scoring a system on a set
# --------------------------------------------------------------------------


def score(rows: Sequence[dict], hyps: Sequence[str]) -> Dict[str, float]:
    assert len(rows) == len(hyps)
    golds = [r["clean"] for r in rows]
    em = sum(exact_match(h, g) for h, g in zip(hyps, golds)) / max(1, len(rows))
    wer = word_error_rate(hyps, golds)
    wer_spoken = word_error_rate_spoken(hyps, golds)

    filler = PRF()
    repairs_seen = 0
    repairs_ok = 0
    for row, hyp in zip(rows, hyps):
        filler.add(filler_removal_counts(row, hyp))
        verdict = repair_resolved(row, hyp)
        if verdict is not None:
            repairs_seen += 1
            repairs_ok += int(verdict)

    return {
        "exact_match": em,
        "wer": wer,
        "wer_spoken": wer_spoken,
        "filler_f1": filler.f1,
        "filler_precision": filler.precision,
        "filler_recall": filler.recall,
        "repair_accuracy": repairs_ok / repairs_seen if repairs_seen else float("nan"),
        "n": len(rows),
        "n_repairs": repairs_seen,
    }


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


def load_model(path: Path, device: Optional[str] = None):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from mend.train import pick_device

    device = device or pick_device()
    path = Path(path)
    adapter = path / "adapter_config.json"
    if adapter.exists():
        from peft import PeftModel

        base = json.loads(adapter.read_text())["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(base)
        model = AutoModelForSeq2SeqLM.from_pretrained(base)
        model = PeftModel.from_pretrained(model, str(path)).merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(path))
    model.to(device).eval()
    torch.set_grad_enabled(False)
    return model, tokenizer, device


def run_model(model, tokenizer, device: str, texts: Sequence[str], batch: int = 32) -> List[str]:
    import torch

    from mend.train import PREFIX

    out: List[str] = []
    for i in range(0, len(texts), batch):
        chunk = [PREFIX + t for t in texts[i : i + batch]]
        enc = tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=192,
            pad_to_multiple_of=8,
        ).to(device)
        gen = model.generate(**enc, max_new_tokens=160, num_beams=1, do_sample=False)
        out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        print(f"  decoded {min(i + batch, len(texts))}/{len(texts)}", end="\r", file=sys.stderr)
        # See the note in train.py: the MPS allocator pools per shape and holds
        # on to everything unless it is told otherwise.
        if device == "mps" and (i // batch) % 20 == 19:
            torch.mps.empty_cache()
    print(file=sys.stderr)
    return out


# --------------------------------------------------------------------------
# table
# --------------------------------------------------------------------------

COLUMNS = [
    ("system", "System"),
    ("dataset", "Set"),
    ("n", "N"),
    ("exact_match", "Exact match"),
    ("wer", "WER"),
    ("wer_spoken", "WER (words only)"),
    ("filler_f1", "Filler removal F1"),
    ("repair_accuracy", "Repair resolution"),
    ("n_repairs", "Repairs scored"),
]


def _fmt(key: str, value) -> str:
    if key in ("system", "dataset"):
        return str(value)
    if key in ("n", "n_repairs"):
        return str(int(value))
    return f"{value:.3f}"


def write_table(results: List[dict], out_dir: Path) -> Tuple[Path, Path]:
    md_path = out_dir / "eval_table.md"
    csv_path = out_dir / "eval_table.csv"

    header = [label for _, label in COLUMNS]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for row in results:
        lines.append("| " + " | ".join(_fmt(k, row[k]) for k, _ in COLUMNS) + " |")
    lines.append("")
    lines.append(
        "Exact match ignores whitespace differences and treats the sentence-final "
        "mark as optional. WER is computed over whitespace tokens with punctuation "
        "attached, so a missing comma counts as one error. Filler removal F1 counts "
        "occurrences of filler words deleted, so a filler word used literally can be "
        "a false positive. Repair resolution asks whether the reparandum is gone and "
        "the repair is present; on the synthetic split it uses the injector spans, "
        "and on the hard set, which has no spans, it uses the word-multiset "
        "difference between input and gold."
    )
    md_path.write_text("\n".join(lines) + "\n")

    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([k for k, _ in COLUMNS] + ["filler_precision", "filler_recall"])
        for row in results:
            writer.writerow(
                [row[k] for k, _ in COLUMNS] + [row["filler_precision"], row["filler_recall"]]
            )
    return md_path, csv_path


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the Mend cleanup model")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--model", type=Path, default=Path("outputs/model"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="cap examples per set")
    ap.add_argument("--swda-sample", dest="swda_sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    sets = {
        "held-out synthetic": read_jsonl(args.data / "test.jsonl"),
        "hard set": read_jsonl(args.data / "hard_set.jsonl"),
    }
    swda_path = args.data / "swda_eval.jsonl"
    if swda_path.exists():
        swda = read_jsonl(swda_path)
        # A fixed sample, so the row is comparable between runs without
        # decoding all 80k utterances every time.
        random.Random(args.seed).shuffle(swda)
        sets["real speech (SwDA)"] = swda[: args.swda_sample]
    else:
        print(f"no {swda_path}; skipping the real-speech row", file=sys.stderr)
    if args.limit:
        sets = {k: v[: args.limit] for k, v in sets.items()}

    model, tokenizer, device = load_model(args.model, args.device)
    print(f"model: {args.model} on {device}", file=sys.stderr)

    results: List[dict] = []
    predictions: List[dict] = []
    for name, rows in sets.items():
        inputs = [r["disfluent"] for r in rows]
        print(f"decoding {name} ({len(rows)})", file=sys.stderr)
        hyps = run_model(model, tokenizer, device, inputs, args.batch)
        base = [copy_baseline(t) for t in inputs]

        for system, outputs in (("Copy baseline", base), ("Mend (flan-t5-small)", hyps)):
            row = {"system": system, "dataset": name}
            row.update(score(rows, outputs))
            results.append(row)

        for r, hyp, b in zip(rows, hyps, base):
            predictions.append(
                {
                    "dataset": name,
                    "disfluent": r["disfluent"],
                    "gold": r["clean"],
                    "mend": hyp,
                    "copy_baseline": b,
                }
            )

    # baseline rows first within each set, which is how the table reads best
    order = {"Copy baseline": 0, "Mend (flan-t5-small)": 1}
    results.sort(key=lambda r: (list(sets).index(r["dataset"]), order[r["system"]]))

    md_path, csv_path = write_table(results, args.out)
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for row in predictions:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(md_path.read_text())
    print(f"wrote {md_path} and {csv_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
