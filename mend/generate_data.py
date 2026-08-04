"""Build (disfluent, clean) training pairs.

Two sources feed the pool of clean sentences:

* WikiText, filtered down to well-formed sentences of 5-40 words. This gives
  general written English with real punctuation.
* Template-generated imperatives in the dictation register ("send the invoice
  to Sarah by Friday"), because that is the domain the model is aimed at and
  WikiText contains almost none of it.

The pool is split into train/validation/test *before* any injection happens, so
no clean sentence appears in more than one split. A fifth of the imperative
templates are withheld from training entirely and contribute only to the test
split, which gives a read on whether the model generalises past the exact
frames it was trained on.

Usage::

    python -m mend.generate_data --smoke
    python -m mend.generate_data --train 50000 --val 2000 --test 2000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from mend.injector import InjectionConfig, canonical_clean_tokens, inject

# --------------------------------------------------------------------------
# template source
# --------------------------------------------------------------------------

SLOTS: Dict[str, Tuple[str, ...]] = {
    "name": (
        "Sarah", "John", "Michael", "Emily", "David", "Rachel", "Daniel",
        "Laura", "Kevin", "Anna", "Peter", "Maria", "Tom", "Julia", "Chris",
        "Nina", "Omar", "Priya", "Hannah", "Marcus",
    ),
    "doc": (
        "report", "invoice", "contract", "summary", "proposal", "agenda",
        "memo", "draft",
    ),
    "day": (
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
        "Sunday",
    ),
    "month": (
        "January", "February", "March", "April", "June", "July", "August",
        "September", "October", "November", "December",
    ),
    "time": ("nine", "ten", "eleven", "twelve", "two", "three", "four", "five"),
    "part_of_day": ("morning", "afternoon", "evening"),
    "room": (
        "office", "kitchen", "garage", "basement", "hallway", "lobby",
        "warehouse", "studio",
    ),
    "colour": ("red", "blue", "green", "black", "white", "yellow", "purple", "orange"),
    "object": ("folder", "binder", "laptop", "box", "envelope", "notebook", "parcel"),
    "channel": ("email", "message", "call", "text", "letter"),
    "count": ("two", "three", "four", "five", "six", "eight", "ten", "twelve"),
    "number": tuple(str(n) for n in (2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 20, 21, 24, 30)),
    "topic": (
        "budget", "launch", "migration", "hiring plan", "renewal", "audit",
        "field trip", "rollout",
    ),
    "place": (
        "the station", "the airport", "the clinic", "the school",
        "the main entrance", "the north gate", "the front desk",
    ),
    "adj": ("large", "small", "quiet", "upstairs", "downstairs", "corner"),
    "status": ("ready", "late", "signed", "approved", "still open", "on hold"),
    "task": (
        "renew the licence", "book the flights", "order more paper",
        "return the keys", "confirm the numbers", "update the slides",
        "chase the supplier", "back up the drive",
    ),
}

TEMPLATES: Tuple[str, ...] = (
    "Send the {doc} to {name} before {day}.",
    "Send {name} the {colour} {object} from the {room}.",
    "Forward the {doc} to {name} and {name2}, please.",
    "Schedule the {topic} review for {day} at {time}.",
    "Schedule a {count}-hour session on {month} {number}.",
    "Remind me to {task} on {day} {part_of_day}.",
    "Remind {name} that the {doc} is {status}.",
    "Tell {name} the {doc} is {status} and needs a second look.",
    "Ask {name} whether the {doc} for the {topic} is {status}.",
    "Book a table for {count} people at {time} on {day}.",
    "Book the {adj} meeting room for {day} {part_of_day}.",
    "Move the {colour} {object} from the {room} to the {room2}.",
    "Move the {topic} meeting to {month} {number}.",
    "Email {name} about the {topic} and copy {name2}.",
    "Reply to {name} and say the {doc} will be ready by {day}.",
    "Call {name} at {time} and confirm the {doc}.",
    "Print {count} copies of the {doc} and leave them in the {room}.",
    "Order {number} more boxes of paper for the {room}.",
    "Cancel the {day} {part_of_day} call with {name}.",
    "Push the {topic} deadline back to {month} {number}.",
    "Add {name} to the {topic} thread and drop {name2} from it.",
    "Pick up the {object} from {place} on {day}.",
    "Meet {name} at {place} at {time} on {day}.",
    "File the signed {doc} under the {topic} and tell {name}.",
    "Draft a short {channel} to {name} about the {topic}.",
    "Confirm with {name} that the {adj} {room} is free on {day}.",
    "Let {name} know the {doc} arrived on {month} {number}.",
    "Check whether {name} sent the {doc} before {day} {part_of_day}.",
    "Bring the {colour} {object} to {place} by {time}.",
    "Split the {doc} into {count} sections and send it to {name}.",
)


def _fill(template: str, rng: random.Random) -> str:
    """Fill one template. ``name2``/``room2`` draw a second, distinct value."""
    out = template
    used: Dict[str, str] = {}
    for match in re.findall(r"\{(\w+)\}", template):
        if match in used:
            continue
        base = match.rstrip("2")
        options = SLOTS[base]
        choice = rng.choice(options)
        if match.endswith("2"):
            alt = [o for o in options if o != used.get(base)]
            choice = rng.choice(alt)
        used[match] = choice
        if not match.endswith("2"):
            used.setdefault(base, choice)
    for key, value in used.items():
        out = out.replace("{" + key + "}", value)
    return out


def template_sentences(
    template_ids: Sequence[int], per_template: int, rng: random.Random
) -> List[Tuple[str, int]]:
    """Generate ``per_template`` distinct sentences for each listed template."""
    out: List[Tuple[str, int]] = []
    for tid in template_ids:
        seen = set()
        attempts = 0
        while len(seen) < per_template and attempts < per_template * 20:
            attempts += 1
            sent = _fill(TEMPLATES[tid], rng)
            if sent not in seen:
                seen.add(sent)
                out.append((sent, tid))
    return out


# --------------------------------------------------------------------------
# wikitext source
# --------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")
_BAD_SUBSTRINGS = ("<unk>", "@-@", "@.@", "@,@", "=", "|", "http", "–", "—")
_ALLOWED = re.compile(r"^[A-Za-z0-9 ,.;:!?'()%&$£€-]+$")
# an apostrophe that is not sitting between two letters, e.g. the dangling
# possessive WikiText writes as "the boys '"
_LOOSE_APOSTROPHE = re.compile(r"(?<![A-Za-z])'|'(?![A-Za-z])")
# titles the sentence splitter will happily mistake for a sentence end
_ABBREVIATIONS = ("St.", "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Jr.", "Sr.", "No.", "vs.")

_SPACE_BEFORE = re.compile(r"\s+([,.;:!?%)\]])")
_SPACE_AFTER = re.compile(r"([(\[$£€])\s+")
_CLITIC = re.compile(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", re.IGNORECASE)


def detokenize(text: str) -> str:
    """Undo WikiText's whitespace tokenisation.

    WikiText separates punctuation from its word ("the report , which ..."),
    which would strand every comma and full stop as its own token. The clean
    side of a training pair has to carry real punctuation, so put it back.
    """
    text = text.replace('"', "").replace("“", "").replace("”", "")
    text = " ".join(text.split())
    text = _CLITIC.sub(r"\1", text)
    text = _SPACE_BEFORE.sub(r"\1", text)
    text = _SPACE_AFTER.sub(r"\1", text)
    return text.strip()


def _clean_sentences_from_line(line: str) -> Iterable[str]:
    line = line.strip()
    if not line or line.startswith("="):
        return
    for raw in _SENT_SPLIT.split(line):
        if any(b in raw for b in _BAD_SUBSTRINGS):
            continue
        sent = detokenize(raw)
        if not sent or not sent[0].isupper() or sent[-1] not in ".!?":
            continue
        words = sent.split()
        if not 5 <= len(words) <= 40:
            continue
        if not _ALLOWED.match(sent):
            continue
        if _LOOSE_APOSTROPHE.search(sent):
            continue
        if sent.endswith(_ABBREVIATIONS):
            continue
        # a sentence made mostly of numerals is not useful written prose
        if sum(w[0].isdigit() for w in words) > len(words) // 3:
            continue
        # every token must survive to the clean target with its spelling intact
        if len(canonical_clean_tokens(sent)) != len(words):
            continue
        yield sent


def load_wikitext_sentences(limit: int, verbose: bool = True) -> List[str]:
    """Pull well-formed sentences from WikiText, largest corpus first."""
    if limit <= 0:
        return []
    try:
        from datasets import load_dataset
    except ImportError:  # pragma: no cover - dependency is declared
        print("datasets is not installed; continuing with templates only", file=sys.stderr)
        return []

    for config in ("wikitext-103-raw-v1", "wikitext-2-raw-v1"):
        try:
            if verbose:
                print(f"loading {config} ...", file=sys.stderr)
            ds = load_dataset("wikitext", config, split="train", streaming=True)
            seen = set()
            out: List[str] = []
            for row in ds:
                for sent in _clean_sentences_from_line(row["text"]):
                    if sent in seen:
                        continue
                    seen.add(sent)
                    out.append(sent)
                    if len(out) >= limit:
                        if verbose:
                            print(f"  {len(out)} sentences from {config}", file=sys.stderr)
                        return out
            if out:
                if verbose:
                    print(
                        f"  {len(out)} sentences from {config} (corpus exhausted)",
                        file=sys.stderr,
                    )
                return out
        except Exception as exc:  # noqa: BLE001 - any download/parse failure
            print(f"  {config} unavailable: {exc}", file=sys.stderr)
    print("no WikiText available; continuing with templates only", file=sys.stderr)
    return []


# --------------------------------------------------------------------------
# assembling the splits
# --------------------------------------------------------------------------


@dataclass
class SourceSentence:
    text: str
    source: str  # "wikitext" | "template" | "template_heldout"
    template_id: Optional[int] = None


@dataclass
class Pools:
    train: List[SourceSentence]
    val: List[SourceSentence]
    test: List[SourceSentence]


def build_pools(
    n_train_pairs: int,
    n_val_pairs: int,
    n_test_pairs: int,
    seed: int,
    heldout_fraction: float = 0.2,
    template_share: float = 0.35,
    verbose: bool = True,
) -> Pools:
    """Split clean sentences into disjoint train/val/test pools."""
    rng = random.Random(seed)
    total_pairs = n_train_pairs + n_val_pairs + n_test_pairs

    # Each source sentence is reused at most twice, with different seeds.
    n_sources = max(200, total_pairs // 2)
    n_template_sources = int(n_sources * template_share)
    n_wiki_sources = n_sources - n_template_sources

    n_heldout = max(1, round(len(TEMPLATES) * heldout_fraction))
    template_ids = list(range(len(TEMPLATES)))
    rng.shuffle(template_ids)
    heldout_ids = sorted(template_ids[:n_heldout])
    train_ids = sorted(template_ids[n_heldout:])
    if verbose:
        print(f"held-out templates: {heldout_ids}", file=sys.stderr)

    per_template = max(1, n_template_sources // max(1, len(train_ids)))
    shared = template_sentences(train_ids, per_template, rng)
    # The held-out templates only need to cover the test split.
    per_heldout = max(1, (n_test_pairs // 4) // max(1, len(heldout_ids)))
    heldout = template_sentences(heldout_ids, per_heldout, rng)

    wiki = load_wikitext_sentences(n_wiki_sources, verbose=verbose)
    if not wiki and verbose:
        print("running with template sentences only", file=sys.stderr)

    rng.shuffle(shared)
    rng.shuffle(wiki)

    # Proportional split of the shared pools, sentence level, no overlap.
    def cut(items: List, weights: Tuple[float, float, float]) -> Tuple[List, List, List]:
        n = len(items)
        a = int(n * weights[0])
        b = a + int(n * weights[1])
        return items[:a], items[a:b], items[b:]

    frac_val = n_val_pairs / max(1, total_pairs)
    frac_test = n_test_pairs / max(1, total_pairs)
    frac_train = 1.0 - frac_val - frac_test
    weights = (frac_train, frac_val, frac_test)

    wiki_tr, wiki_va, wiki_te = cut(wiki, weights)
    tpl_tr, tpl_va, tpl_te = cut(shared, weights)

    def wrap(texts, source, with_id=False):
        if with_id:
            return [SourceSentence(t, source, tid) for t, tid in texts]
        return [SourceSentence(t, source) for t in texts]

    train = wrap(wiki_tr, "wikitext") + wrap(tpl_tr, "template", True)
    val = wrap(wiki_va, "wikitext") + wrap(tpl_va, "template", True)
    test = (
        wrap(wiki_te, "wikitext")
        + wrap(tpl_te, "template", True)
        + wrap(heldout, "template_heldout", True)
    )
    for pool in (train, val, test):
        rng.shuffle(pool)
    return Pools(train, val, test)


def make_pairs(
    pool: Sequence[SourceSentence],
    n_pairs: int,
    config: InjectionConfig,
    base_seed: int,
) -> List[dict]:
    """Inject disfluencies, cycling the pool with a fresh seed each time."""
    records: List[dict] = []
    if not pool:
        return records
    i = 0
    while len(records) < n_pairs:
        src = pool[i % len(pool)]
        result = inject(src.text, config, rng_seed=base_seed + i)
        i += 1
        if not result.clean_text:
            continue
        row = result.to_dict()
        row["source"] = src.source
        row["template_id"] = src.template_id
        records.append(row)
    return records


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> List[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def summarise(rows: Sequence[dict]) -> str:
    counts: Dict[str, int] = {}
    for row in rows:
        for ann in row["annotations"]:
            counts[ann["type"]] = counts.get(ann["type"], 0) + 1
    with_any = sum(1 for r in rows if r["annotations"])
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    return f"{len(rows)} pairs, {with_any} with a disfluency, " + " ".join(parts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate (disfluent, clean) pairs")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--train", type=int, default=50_000)
    ap.add_argument("--val", type=int, default=2_000)
    ap.add_argument("--test", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="generate 100 pairs from templates only and print a sample",
    )
    args = ap.parse_args(argv)

    config = InjectionConfig()

    if args.smoke:
        pools = build_pools(80, 10, 10, args.seed, template_share=1.0, verbose=True)
        rows = make_pairs(pools.train, 80, config, args.seed)
        rows += make_pairs(pools.val, 10, config, args.seed + 10_000)
        rows += make_pairs(pools.test, 10, config, args.seed + 20_000)
        path = args.out / "smoke.jsonl"
        write_jsonl(path, rows)
        print(f"wrote {len(rows)} pairs to {path}")
        print(summarise(rows))
        print()
        for row in rows[:10]:
            print(" IN:", row["disfluent"])
            print("OUT:", row["clean"])
            print("    ", [a["type"] for a in row["annotations"]])
            print()
        return 0

    pools = build_pools(args.train, args.val, args.test, args.seed)
    print(
        f"source sentences: train={len(pools.train)} val={len(pools.val)} "
        f"test={len(pools.test)}",
        file=sys.stderr,
    )
    for name, pool, n, offset in (
        ("train", pools.train, args.train, 0),
        ("val", pools.val, args.val, 1_000_000),
        ("test", pools.test, args.test, 2_000_000),
    ):
        rows = make_pairs(pool, n, config, args.seed + offset)
        written = write_jsonl(args.out / f"{name}.jsonl", rows)
        print(f"{name}: {summarise(rows)} -> {args.out / (name + '.jsonl')} ({written})")
    return 0


if __name__ == "__main__":
    code = main()
    # The streaming WikiText reader leaves a live worker thread behind, which
    # keeps the interpreter alive long after the files are written. Everything
    # is flushed by this point, so leave immediately.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
