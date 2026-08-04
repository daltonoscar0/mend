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
repair spans in exactly the schema :mod:`mend.injector` produces. No model or
data-generation choice in this repo was made by looking at it.

Get the corpus (14 MB, no licence gate)::

    curl -L -o swda.zip https://github.com/cgpotts/swda/raw/master/swda.zip
    unzip -q swda.zip -d swda

Then::

    python -m mend.swda --root swda --out data/swda_eval.jsonl
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
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
    root: Path, min_words: int = 5, max_words: int = 40, keep_fragments: bool = False
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
                disfluent = " ".join(parsed.disfluent_tokens)
                if not keep_fragments and re.search(r"\b\w+-(\s|$)", disfluent + " "):
                    # a cut-off word like "d-"; nothing in training resembles it
                    stats["fragment"] += 1
                    continue
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build a real-speech evaluation set from SwDA")
    ap.add_argument("--root", type=Path, required=True, help="unzipped swda directory")
    ap.add_argument("--out", type=Path, default=Path("data/swda_eval.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-fragments", dest="keep_fragments", action="store_true")
    args = ap.parse_args(argv)

    rows = load_utterances(args.root, keep_fragments=args.keep_fragments)
    if args.limit:
        rows = rows[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    types: Counter = Counter()
    for row in rows:
        for ann in row["annotations"]:
            types[ann["type"]] += 1
    print(f"wrote {len(rows)} utterances to {args.out}")
    print("annotations: " + "  ".join(f"{k}={v}" for k, v in sorted(types.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
