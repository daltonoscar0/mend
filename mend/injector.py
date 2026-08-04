"""Disfluency injector.

Takes clean written text and produces a spoken-style rendering of it, in the
structure Shriberg uses for self-repairs:

    send it to  john   no wait     sarah
                ^^^^   ^^^^^^^     ^^^^^
                reparandum         repair
                       interregnum

Every injected disfluency is recorded with its span offsets on the *disfluent*
side, so the reparandum / interregnum / repair boundaries stay recoverable for
downstream analysis. The disfluent side is also lowercased and stripped of
punctuation, which mimics raw ASR output and gives the cleanup model a
punctuation-restoration job on top of the disfluency job.

Injected material is always removable: deleting the annotated reparandum and
interregnum spans and mapping what is left back through ``alignment`` returns
the clean token sequence exactly.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# lexicons
# --------------------------------------------------------------------------

FILLERS: Tuple[Tuple[str, ...], ...] = (
    ("um",),
    ("uh",),
    ("like",),
    ("you", "know"),
)

INTERREGNA: Tuple[Tuple[str, ...], ...] = (
    ("no", "wait"),
    ("i", "mean"),
    ("sorry",),
    ("no",),
)

FALSE_START_FRAGMENTS: Tuple[Tuple[str, ...], ...] = (
    ("can", "you"),
    ("could", "you"),
    ("i", "want", "to"),
    ("i", "need", "to"),
    ("we", "should"),
    ("do", "you"),
    ("let", "me"),
    ("i", "was", "going", "to"),
)

# Swap groups for repairs. A reparandum is drawn from the same group as the
# word it stands in for, which is what makes the wrong version plausible.
SWAP_GROUPS: Tuple[Tuple[str, ...], ...] = (
    # people
    (
        "john", "sarah", "michael", "emily", "david", "rachel", "daniel",
        "laura", "kevin", "anna", "peter", "maria", "tom", "julia", "chris",
        "nina", "omar", "priya", "hannah", "marcus",
    ),
    # days
    ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"),
    # months
    (
        "january", "february", "march", "april", "june", "july",
        "august", "september", "october", "november", "december",
    ),
    # number words
    (
        "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "twenty", "thirty", "fifty",
    ),
    # ordinals
    ("first", "second", "third", "fourth", "fifth"),
    # colours
    ("red", "blue", "green", "black", "white", "yellow", "purple", "orange", "grey"),
    # places around a building
    ("office", "kitchen", "garage", "basement", "hallway", "lobby", "warehouse", "studio"),
    # documents
    ("report", "invoice", "contract", "summary", "proposal", "agenda", "memo", "draft"),
    # time of day
    ("morning", "afternoon", "evening", "tonight", "tomorrow"),
    # directions
    ("north", "south", "east", "west"),
    # channels
    ("email", "message", "call", "text", "letter"),
)

_SWAP_INDEX: Dict[str, int] = {}
for _gi, _group in enumerate(SWAP_GROUPS):
    for _w in _group:
        _SWAP_INDEX[_w] = _gi

STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were
    be been being do does did have has had i you he she it we they me him her us
    them my your his its our their this that these those not no so then than
    there here about into over under up down out off just very can could would
    should will shall may might must""".split()
)

_STRIP_CHARS = " \t\n.,;:!?\"'()[]{}<>—–-…‘’“”"
_QUOTE_TABLE = str.maketrans("", "", "\"“”«»")
_HAS_ALNUM = re.compile(r"[0-9a-zA-Z]")


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    """Half-open token span ``[start, end)`` on the disfluent side."""

    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start

    def indices(self) -> range:
        return range(self.start, self.end)


@dataclass
class Annotation:
    """One injected disfluency.

    ``reparandum`` is the material the speaker abandons, ``interregnum`` the
    editing phrase between the two, ``repair`` the material that replaces the
    reparandum. Types that lack a component leave it as ``None``: a filled
    pause is interregnum only, a repetition has no interregnum.
    """

    type: str
    reparandum: Optional[Span] = None
    interregnum: Optional[Span] = None
    repair: Optional[Span] = None

    def deleted_indices(self) -> List[int]:
        """Disfluent token positions that should not survive cleanup."""
        out: List[int] = []
        for span in (self.reparandum, self.interregnum):
            if span is not None:
                out.extend(span.indices())
        return out

    def boundary_index(self) -> Optional[int]:
        """The token at which a listener learns something went wrong.

        That is the start of the interregnum when there is one, otherwise the
        start of the repair, otherwise the start of the reparandum.
        """
        for span in (self.interregnum, self.repair, self.reparandum):
            if span is not None:
                return span.start
        return None


@dataclass
class InjectionResult:
    disfluent_text: str
    clean_text: str
    disfluent_tokens: List[str] = field(default_factory=list)
    clean_tokens: List[str] = field(default_factory=list)
    # alignment[i] is the clean-token index that disfluent token i came from,
    # or None when the token was injected.
    alignment: List[Optional[int]] = field(default_factory=list)
    annotations: List[Annotation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "disfluent": self.disfluent_text,
            "clean": self.clean_text,
            "disfluent_tokens": list(self.disfluent_tokens),
            "clean_tokens": list(self.clean_tokens),
            "alignment": list(self.alignment),
            "annotations": [asdict(a) for a in self.annotations],
        }

    @staticmethod
    def from_dict(d: dict) -> "InjectionResult":
        def _span(x) -> Optional[Span]:
            return None if x is None else Span(x["start"], x["end"])

        anns = [
            Annotation(
                type=a["type"],
                reparandum=_span(a.get("reparandum")),
                interregnum=_span(a.get("interregnum")),
                repair=_span(a.get("repair")),
            )
            for a in d.get("annotations", [])
        ]
        return InjectionResult(
            disfluent_text=d["disfluent"],
            clean_text=d["clean"],
            disfluent_tokens=list(d.get("disfluent_tokens", [])),
            clean_tokens=list(d.get("clean_tokens", [])),
            alignment=list(d.get("alignment", [])),
            annotations=anns,
        )


@dataclass
class InjectionConfig:
    """Per-sentence probabilities of injecting each disfluency type.

    ``filler_rate`` is the chance of a first filled pause; each further filler
    is drawn with the same probability, capped at ``max_fillers``.
    """

    filler_rate: float = 0.45
    repetition_rate: float = 0.18
    repair_rate: float = 0.30
    false_start_rate: float = 0.10
    # given a repair, how often an editing phrase separates the two parts
    interregnum_rate: float = 0.65
    max_fillers: int = 3
    repetition_bigram_rate: float = 0.25
    lowercase: bool = True
    strip_punctuation: bool = True

    @staticmethod
    def from_dict(d: Optional[dict]) -> "InjectionConfig":
        if not d:
            return InjectionConfig()
        known = set(InjectionConfig.__dataclass_fields__)
        return InjectionConfig(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------
# tokenisation helpers
# --------------------------------------------------------------------------


def tokenize(text: str) -> List[str]:
    """Whitespace tokenisation, keeping punctuation attached to its word."""
    return text.split()


def spoken_form(token: str, cfg: Optional[InjectionConfig] = None) -> str:
    """Render one written token the way an ASR transcript would show it."""
    cfg = cfg or InjectionConfig()
    out = token.translate(_QUOTE_TABLE)
    if cfg.strip_punctuation:
        out = out.strip(_STRIP_CHARS)
    if cfg.lowercase:
        out = out.lower()
    return out


def canonical_clean_tokens(text: str, cfg: Optional[InjectionConfig] = None) -> List[str]:
    """Clean tokens that survive to the target side.

    Tokens with no alphanumeric content (a stray em dash, say) have no spoken
    form, so they are dropped and the clean target is rebuilt without them.
    """
    return [t for t in tokenize(text) if _HAS_ALNUM.search(t)]


def _is_content_word(token: str) -> bool:
    stripped = token.strip(_STRIP_CHARS).lower()
    return len(stripped) > 3 and stripped not in STOPWORDS


def _ends_clause(token: str) -> bool:
    return token.rstrip("\"”'’)").endswith((",", ";", ":"))


# --------------------------------------------------------------------------
# repair material
# --------------------------------------------------------------------------


def _alternative_for(word: str, rng: random.Random) -> Optional[str]:
    """A wrong-but-plausible stand-in drawn from the same category."""
    key = word.strip(_STRIP_CHARS).lower()
    if not key:
        return None
    gi = _SWAP_INDEX.get(key)
    if gi is not None:
        options = [w for w in SWAP_GROUPS[gi] if w != key]
        return rng.choice(options) if options else None
    if key.isdigit():
        n = int(key)
        deltas = [d for d in (-3, -2, -1, 1, 2, 3, 10, -10) if n + d > 0]
        if not deltas:
            return None
        return str(n + rng.choice(deltas))
    return None


def _swappable_positions(clean_tokens: Sequence[str]) -> List[int]:
    out = []
    for i, tok in enumerate(clean_tokens):
        key = tok.strip(_STRIP_CHARS).lower()
        if key in _SWAP_INDEX or key.isdigit():
            out.append(i)
    return out


# --------------------------------------------------------------------------
# the injector
# --------------------------------------------------------------------------


def _plan_edits(
    clean_tokens: Sequence[str], cfg: InjectionConfig, rng: random.Random
) -> Tuple[List[dict], Optional[Tuple[str, ...]]]:
    """Choose non-overlapping edits over clean-token positions."""
    n = len(clean_tokens)
    edits: List[dict] = []
    reserved: set = set()

    fragment: Optional[Tuple[str, ...]] = None
    if n >= 6 and rng.random() < cfg.false_start_rate:
        first_two = " ".join(t.strip(_STRIP_CHARS).lower() for t in clean_tokens[:2])
        options = [f for f in FALSE_START_FRAGMENTS if " ".join(f) != first_two]
        fragment = rng.choice(options)
        # keep the first few positions clear so the restart resumes cleanly
        reserved.update(range(min(3, n)))

    if rng.random() < cfg.repair_rate:
        candidates = [p for p in _swappable_positions(clean_tokens) if p not in reserved]
        if candidates:
            pos = rng.choice(candidates)
            wrong = _alternative_for(clean_tokens[pos], rng)
            if wrong is not None:
                interregnum = (
                    rng.choice(INTERREGNA) if rng.random() < cfg.interregnum_rate else None
                )
                edits.append(
                    {"kind": "repair", "pos": pos, "wrong": wrong, "interregnum": interregnum}
                )
                reserved.add(pos)

    if n >= 3 and rng.random() < cfg.repetition_rate:
        length = 2 if (n >= 5 and rng.random() < cfg.repetition_bigram_rate) else 1
        candidates = [
            p
            for p in range(n - length + 1)
            if not any(q in reserved for q in range(p, p + length))
        ]
        if candidates:
            pos = rng.choice(candidates)
            edits.append({"kind": "repetition", "pos": pos, "length": length})
            reserved.update(range(pos, pos + length))

    # filled pauses, inserted *before* a clean token
    slots = [
        p
        for p in range(n)
        if p not in reserved
        and (p == 0 or _ends_clause(clean_tokens[p - 1]) or _is_content_word(clean_tokens[p]))
    ]
    rng.shuffle(slots)
    taken = 0
    for _ in range(cfg.max_fillers):
        if rng.random() >= cfg.filler_rate or taken >= len(slots):
            break
        edits.append({"kind": "filler", "pos": slots[taken], "words": rng.choice(FILLERS)})
        taken += 1

    order = {"filler": 0, "repair": 1, "repetition": 1}
    edits.sort(key=lambda e: (e["pos"], order[e["kind"]]))
    return edits, fragment


def inject(
    clean_text: str,
    config: Optional[InjectionConfig] = None,
    rng_seed: int = 0,
) -> InjectionResult:
    """Render ``clean_text`` as disfluent speech with annotated repair spans."""
    cfg = config or InjectionConfig()
    rng = random.Random(rng_seed)

    clean_tokens = canonical_clean_tokens(clean_text, cfg)
    if not clean_tokens:
        return InjectionResult(disfluent_text="", clean_text="")

    spoken = [spoken_form(t, cfg) for t in clean_tokens]
    edits, fragment = _plan_edits(clean_tokens, cfg, rng)
    by_pos: Dict[int, List[dict]] = {}
    for e in edits:
        by_pos.setdefault(e["pos"], []).append(e)

    out: List[str] = []
    align: List[Optional[int]] = []
    anns: List[Annotation] = []

    def emit(token: str, source: Optional[int]) -> None:
        out.append(token)
        align.append(source)

    if fragment is not None:
        start = len(out)
        for w in fragment:
            emit(w, None)
        anns.append(
            Annotation(
                type="false_start",
                reparandum=Span(start, len(out)),
                # the restart resumes on clean token 0, emitted immediately after
                repair=Span(len(out), len(out) + min(3, len(clean_tokens))),
            )
        )

    for i, tok in enumerate(spoken):
        for e in by_pos.get(i, []):
            if e["kind"] == "filler":
                start = len(out)
                for w in e["words"]:
                    emit(w, None)
                anns.append(Annotation(type="filler", interregnum=Span(start, len(out))))
            elif e["kind"] == "repair":
                rep_start = len(out)
                emit(e["wrong"], None)
                rep_end = len(out)
                inter_span = None
                if e["interregnum"] is not None:
                    i_start = len(out)
                    for w in e["interregnum"]:
                        emit(w, None)
                    inter_span = Span(i_start, len(out))
                anns.append(
                    Annotation(
                        type="repair",
                        reparandum=Span(rep_start, rep_end),
                        interregnum=inter_span,
                        repair=Span(len(out), len(out) + 1),
                    )
                )
            elif e["kind"] == "repetition":
                rep_start = len(out)
                for k in range(e["length"]):
                    emit(spoken[i + k], None)
                rep_end = len(out)
                anns.append(
                    Annotation(
                        type="repetition",
                        reparandum=Span(rep_start, rep_end),
                        repair=Span(rep_end, rep_end + e["length"]),
                    )
                )
        emit(tok, i)

    return InjectionResult(
        disfluent_text=" ".join(out),
        clean_text=" ".join(clean_tokens),
        disfluent_tokens=out,
        clean_tokens=clean_tokens,
        alignment=align,
        annotations=anns,
    )


# --------------------------------------------------------------------------
# inverse, used by the tests and by the evaluation code
# --------------------------------------------------------------------------


def remove_disfluencies(result: InjectionResult) -> str:
    """Delete annotated reparandum/interregnum spans and restore the writing.

    This is the inverse of :func:`inject`: it reproduces ``result.clean_text``
    exactly, casing and punctuation included.
    """
    drop = set()
    for ann in result.annotations:
        drop.update(ann.deleted_indices())
    kept = [
        result.alignment[i]
        for i in range(len(result.disfluent_tokens))
        if i not in drop and result.alignment[i] is not None
    ]
    return " ".join(result.clean_tokens[j] for j in kept)
