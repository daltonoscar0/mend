"""Real disfluent speech, from the Switchboard Dialog Act corpus.

Everything else in Mend is synthetic: the injector decides where disfluencies
go, and the model is then measured on its ability to undo that same injector.
That measures the inverse of one generator, not the ability to clean up speech.

SwDA is the antidote. It is freely redistributable, it is real telephone
conversation, and its transcripts carry the Treebank-3 disfluency annotation,
which is the same structure the injector imitates:

    [ do you, + do you ] have any experience    repair, reparandum then repair
    {F uh, }                                    filled pause
    {E I mean, }                                explicit editing term
    {D you know, }                              discourse marker
    {C and }                                    coordinating conjunction

so a parse gives (disfluent, clean) pairs with reparandum / interregnum /
repair spans in exactly the schema :mod:`mend.injector` produces.

The corpus feeds training as well as evaluation, because synthetic repairs alone
turned out not to teach a model to resolve real ones. It is therefore split by
conversation: the evaluation conversations are never trained on, but the corpus
as a whole no longer has the stronger property it had when it was first added,
that nothing in the repo had ever been fitted to any part of it.

Get the corpus (14 MB, no licence gate)::

    curl -L -o swda.zip https://github.com/cgpotts/swda/raw/master/swda.zip
    unzip -q swda.zip -d swda

Then::

    python -m mend.swda --root swda --out data/swda_all.jsonl --split

which writes the whole parse to ``--out`` for the surprisal analysis, and a
conversation-disjoint ``swda_train`` / ``swda_val`` / ``swda_eval`` beside it.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from mend.injector import Annotation, InjectionResult, Span, spoken_form

# Curly-brace groups whose contents are disfluency rather than content.
# {C ...} is a coordinating conjunction and {A ...} an aside; both are things
# the speaker meant to say, so they stay.
DROP_LABELS = frozenset("FED")

_NONVERBAL = re.compile(r"<<[^>]*>>|<[^>]*>")
_UNCERTAIN = re.compile(r"\(\(|\)\)")
_TOKEN = re.compile(r"\{[A-Z]|\}|\[|\]|\+|[^\s\[\]{}+]+")
_WORD_OK = re.compile(r"^[A-Za-z0-9',.?!;:$%&/-]+$")
_STRIP_EDGE = " ,.?!;:-"


@dataclass
class Utterance:
    disfluent_tokens: List[str]
    clean_tokens: List[str]
    alignment: List[Optional[int]]
    annotations: List[Annotation]


def _pre_clean(text: str) -> str:
    """Strip transcription furniture that is not part of the words."""
    text = _NONVERBAL.sub(" ", text)
    text = _UNCERTAIN.sub(" ", text)
    text = text.replace("#", " ")
    return " ".join(text.split())


def _is_complete(text: str) -> bool:
    """Only whole slash units, never a turn continued from somewhere else."""
    stripped = text.strip()
    if not stripped.endswith("/") or stripped.endswith("-/"):
        return False
    if stripped.startswith("--") or "--" in stripped:
        return False
    return stripped.count("[") == stripped.count("]") and stripped.count("{") == stripped.count("}")


def parse_utterance(text: str) -> Optional[Utterance]:
    """Turn one annotated SwDA utterance into a (disfluent, clean) pair.

    Walks the markup with a stack so that nesting works: an inner repair
    sitting inside an outer reparandum is deleted along with everything else
    in that reparandum.
    """
    if not _is_complete(text):
        return None
    body = _pre_clean(text.strip().rstrip("/").strip())
    if not body:
        return None

    # frames: repair frames carry a mode, group frames carry a drop flag
    repair_stack: List[dict] = []
    group_stack: List[bool] = []
    words: List[Tuple[str, bool]] = []  # (word, dropped)
    repairs: List[dict] = []

    def dropping() -> bool:
        if any(group_stack):
            return True
        return any(f["mode"] == "reparandum" for f in repair_stack)

    for match in _TOKEN.finditer(body):
        tok = match.group(0)
        if tok.startswith("{") and len(tok) == 2:
            group_stack.append(tok[1] in DROP_LABELS)
        elif tok == "}":
            if not group_stack:
                return None
            group_stack.pop()
        elif tok == "[":
            frame = {"mode": "reparandum", "reparandum": [], "repair": [], "editing": []}
            repair_stack.append(frame)
        elif tok == "+":
            if not repair_stack:
                return None
            repair_stack[-1]["mode"] = "repair"
        elif tok == "]":
            if not repair_stack:
                return None
            repairs.append(repair_stack.pop())
        else:
            word = tok.strip(_STRIP_EDGE)
            if not word or not _WORD_OK.match(tok):
                continue
            index = len(words)
            dropped = dropping()
            words.append((tok, dropped))
            for frame in repair_stack:
                if frame["mode"] == "reparandum":
                    frame["reparandum"].append(index)
                elif any(group_stack):
                    # an editing term sits between the two halves
                    frame["editing"].append(index)
                else:
                    frame["repair"].append(index)
            if not repair_stack and any(group_stack):
                words[index] = (tok, True)

    if repair_stack or group_stack:
        return None

    # Build the two sides. The disfluent side is lowercased and stripped of
    # punctuation, the way the model sees ASR output; the clean side keeps the
    # transcriber's casing and punctuation.
    disfluent_tokens: List[str] = []
    clean_tokens: List[str] = []
    alignment: List[Optional[int]] = []
    position: Dict[int, int] = {}

    for index, (raw, dropped) in enumerate(words):
        spoken = spoken_form(raw)
        if not spoken:
            continue
        position[index] = len(disfluent_tokens)
        disfluent_tokens.append(spoken)
        if dropped:
            alignment.append(None)
        else:
            alignment.append(len(clean_tokens))
            clean_tokens.append(raw)

    if not clean_tokens or not disfluent_tokens:
        return None

    def span_of(indices: Sequence[int]) -> Optional[Span]:
        kept = [position[i] for i in indices if i in position]
        if not kept:
            return None
        return Span(min(kept), max(kept) + 1)

    annotations: List[Annotation] = []
    for frame in repairs:
        reparandum = span_of(frame["reparandum"])
        repair = span_of(frame["repair"])
        editing = span_of(frame["editing"])
        if reparandum is None:
            continue
        if repair is None:
            # the speaker dropped the thread instead of correcting it
            annotations.append(
                Annotation(type="false_start", reparandum=reparandum, interregnum=editing)
            )
            continue
        left = [disfluent_tokens[i] for i in reparandum.indices()]
        right = [disfluent_tokens[i] for i in repair.indices()]
        annotations.append(
            Annotation(
                type="repetition" if left == right else "repair",
                reparandum=reparandum,
                interregnum=editing,
                repair=repair,
            )
        )

    # Filled pauses and discourse markers outside any repair.
    claimed = set()
    for ann in annotations:
        for span in (ann.reparandum, ann.interregnum, ann.repair):
            if span is not None:
                claimed.update(span.indices())
    run: List[int] = []
    for i, source in enumerate(alignment):
        if source is None and i not in claimed:
            run.append(i)
        elif run:
            annotations.append(
                Annotation(type="filler", interregnum=Span(run[0], run[-1] + 1))
            )
            run = []
    if run:
        annotations.append(Annotation(type="filler", interregnum=Span(run[0], run[-1] + 1)))

    annotations.sort(key=lambda a: a.boundary_index() or 0)
    return Utterance(disfluent_tokens, clean_tokens, alignment, annotations)


def tidy_clean(tokens: Sequence[str]) -> List[str]:
    """Make the surviving words read as a written sentence.

    Deleting a reparandum can strand a comma or leave the sentence starting
    mid-clause, so trim the edges, capitalise the opening word and make sure
    there is a final stop.
    """
    out = [t for t in tokens if t.strip(_STRIP_EDGE)]
    if not out:
        return []
    out = list(out)
    out[0] = out[0].lstrip(",;:- ")
    if not out[0]:
        return []
    out[0] = out[0][0].upper() + out[0][1:]
    last = out[-1].rstrip(",;:- ")
    if not last:
        return []
    if last[-1] not in ".?!":
        last += "."
    out[-1] = last
    return out


def load_utterances(
    root: Path, min_words: int = 5, max_words: int = 40
) -> List[dict]:
    """Parse every SwDA transcript under ``root`` into evaluation rows."""
    files = sorted(f for f in glob.glob(str(root / "**" / "*.csv"), recursive=True))
    rows: List[dict] = []
    seen = set()
    stats: Counter = Counter()
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "text" not in reader.fieldnames:
                continue
            for record in reader:
                stats["utterances"] += 1
                parsed = parse_utterance(record["text"])
                if parsed is None:
                    stats["unparsed"] += 1
                    continue
                clean = tidy_clean(parsed.clean_tokens)
                if len(clean) != len(parsed.clean_tokens):
                    stats["untidy"] += 1
                    continue
                if not min_words <= len(clean) <= max_words:
                    stats["length"] += 1
                    continue
                # Cut-off words ("d-", "ma-") are NOT filtered here, though an
                # earlier version of this file claimed to. `spoken_form` strips
                # the trailing hyphen while building the tokens, so by this
                # point there is none left to match and the check never fired on
                # any of the 81,107 parsed utterances. They therefore reach
                # training and evaluation as bare words -- "ma-" as "ma" -- which
                # is why the model leaves a stranded fragment in place rather
                # than deleting it: nothing marks it as a fragment.
                disfluent = " ".join(parsed.disfluent_tokens)
                if disfluent in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(disfluent)
                result = InjectionResult(
                    disfluent_text=disfluent,
                    clean_text=" ".join(clean),
                    disfluent_tokens=parsed.disfluent_tokens,
                    clean_tokens=clean,
                    alignment=parsed.alignment,
                    annotations=parsed.annotations,
                )
                row = result.to_dict()
                row["source"] = "swda"
                row["conversation"] = record.get("conversation_no")
                rows.append(row)
                stats["kept"] += 1
    print(
        "  " + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())),
        file=sys.stderr,
    )
    return rows


def split_by_conversation(
    rows: Sequence[dict],
    seed: int = 13,
    val_fraction: float = 0.05,
    eval_fraction: float = 0.25,
) -> Dict[str, List[dict]]:
    """Divide the utterances into train/val/eval along conversation lines.

    Splitting by utterance would leak: the two speakers in a Switchboard call
    talk about one assigned topic for ten minutes, so utterances from the same
    conversation share vocabulary, subject matter and speaker habits. A model
    trained on half a conversation and tested on the other half is being asked
    an easier question than the one we mean to ask.

    Whole conversations therefore go to exactly one split. Conversations vary
    in length (10 to 181 utterances), so the fractions are filled by utterance
    count rather than by conversation count, and land within a conversation's
    length of the target.
    """
    by_conversation: Dict[str, List[dict]] = {}
    for row in rows:
        by_conversation.setdefault(str(row.get("conversation")), []).append(row)

    ids = sorted(by_conversation)
    random.Random(seed).shuffle(ids)

    total = len(rows)
    want = {"eval": int(total * eval_fraction), "val": int(total * val_fraction)}
    out: Dict[str, List[dict]] = {"train": [], "val": [], "eval": []}
    conversations: Dict[str, int] = {"train": 0, "val": 0, "eval": 0}

    # Fill the two held-out splits first, then everything else is training.
    for name in ("eval", "val"):
        for cid in list(ids):
            if len(out[name]) >= want[name]:
                break
            out[name].extend(by_conversation[cid])
            conversations[name] += 1
            ids.remove(cid)
    for cid in ids:
        out["train"].extend(by_conversation[cid])
        conversations["train"] += 1

    for name, split in out.items():
        repairs = sum(
            1 for r in split for a in r["annotations"] if a["type"] == "repair"
        )
        print(
            f"  {name}: {len(split)} utterances from {conversations[name]} "
            f"conversations, {repairs} repairs",
            file=sys.stderr,
        )
    return out


def _write(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read(path: Path) -> List[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build real-speech sets from SwDA")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path, help="unzipped swda directory")
    src.add_argument(
        "--from-jsonl",
        dest="from_jsonl",
        type=Path,
        help="re-split an earlier parse instead of reading the corpus again",
    )
    ap.add_argument("--out", type=Path, default=Path("data/swda_all.jsonl"))
    ap.add_argument(
        "--split",
        action="store_true",
        help="write conversation-disjoint train/val/eval sets next to --out",
    )
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--val-fraction", dest="val_fraction", type=float, default=0.05)
    ap.add_argument("--eval-fraction", dest="eval_fraction", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    if args.from_jsonl:
        rows = _read(args.from_jsonl)
        print(f"read {len(rows)} utterances from {args.from_jsonl}", file=sys.stderr)
    else:
        rows = load_utterances(args.root)
    if args.limit:
        rows = rows[: args.limit]

    def report(name: str, split: Sequence[dict], path: Path) -> None:
        types: Counter = Counter()
        for row in split:
            for ann in row["annotations"]:
                types[ann["type"]] += 1
        print(f"wrote {len(split)} utterances to {path}")
        print("  annotations: " + "  ".join(f"{k}={v}" for k, v in sorted(types.items())))

    # The full parse is always written. The surprisal analysis reads it rather
    # than the held-out split: it measures a property of human speech with a
    # separate language model, so no split of ours is relevant to it, and it
    # wants every repair it can get.
    reading_from_out = args.from_jsonl and args.from_jsonl.resolve() == args.out.resolve()
    if not reading_from_out:
        _write(args.out, rows)
        report("all", rows, args.out)
    if not args.split:
        return 0

    splits = split_by_conversation(
        rows, args.seed, args.val_fraction, args.eval_fraction
    )
    stem = args.out.parent
    for name in ("train", "val", "eval"):
        path = stem / f"swda_{name}.jsonl"
        _write(path, splits[name])
        report(name, splits[name], path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
