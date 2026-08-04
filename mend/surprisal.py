"""Surprisal around repair boundaries.

The claim under test is the standard one about self-repair: the point where a
speaker breaks off and corrects themselves is where a listener's expectations
are violated hardest. If that is true of a language model too, per-token
surprisal should spike at the interruption point.

Per-token surprisal comes from a small causal LM (pythia-160m). Subword
surprisals are summed within a whitespace word, which is the usual way of
turning subword surprisal into word surprisal. Each disfluency is centred on
its boundary token: the first token of the interregnum when there is one,
otherwise the first token of the repair.

The control is the same sentence with the disfluency taken out, measured at the
matching word position, so the comparison is between the same words in the same
sentence with and without the interruption.

Usage::

    python -m mend.surprisal
    python -m mend.surprisal --limit 300     # quick pass
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mend.generate_data import read_jsonl
from mend.injector import InjectionResult, spoken_form

DEFAULT_LM = "EleutherAI/pythia-160m"
WINDOW = 5
TYPES = ("repair", "repetition", "filler")

# Okabe-Ito, validated for deuteranopia/protanopia separation.
COLOURS = {
    "repair": "#0072B2",
    "repetition": "#D55E00",
    "filler": "#009E73",
}
LABELS = {
    "repair": "Repair",
    "repetition": "Repetition",
    "filler": "Filled pause",
}
BASELINE_COLOUR = "#4D4D4D"


# --------------------------------------------------------------------------
# surprisal
# --------------------------------------------------------------------------


class Surprisal:
    """Word-level surprisal in bits from a causal LM."""

    def __init__(self, model_name: str = DEFAULT_LM, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from mend.train import pick_device

        self.torch = torch
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device).eval()
        torch.set_grad_enabled(False)
        self.bos = self.tokenizer.eos_token_id or 0

    def word_surprisals(self, sentences: Sequence[str], batch: int = 16) -> List[np.ndarray]:
        """One array of per-word surprisals (bits) per sentence."""
        torch = self.torch
        out: List[np.ndarray] = []
        for start in range(0, len(sentences), batch):
            chunk = list(sentences[start : start + batch])
            encoded = [
                self.tokenizer(text, return_offsets_mapping=True, truncation=True,
                               max_length=256)
                for text in chunk
            ]
            lengths = [len(e["input_ids"]) for e in encoded]
            width = max(lengths) + 1  # room for the primed first position
            ids = torch.full((len(chunk), width), self.bos, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for i, enc in enumerate(encoded):
                n = len(enc["input_ids"])
                ids[i, 1 : n + 1] = torch.tensor(enc["input_ids"], dtype=torch.long)
                mask[i, : n + 1] = 1
            ids = ids.to(self.device)
            mask = mask.to(self.device)

            logits = self.model(input_ids=ids, attention_mask=mask).logits.float()
            logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
            targets = ids[:, 1:]
            token_bits = -(
                logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) / math.log(2)
            )
            token_bits = token_bits.cpu().numpy()

            for i, (text, enc) in enumerate(zip(chunk, encoded)):
                bits = token_bits[i, : lengths[i]]
                out.append(_fold_to_words(text, enc["offset_mapping"], bits))
        return out


def _word_starts(text: str) -> List[Tuple[int, int]]:
    """Character spans of the whitespace-separated words."""
    spans = []
    pos = 0
    for word in text.split(" "):
        if word:
            spans.append((pos, pos + len(word)))
        pos += len(word) + 1
    return spans


def _fold_to_words(
    text: str, offsets: Sequence[Tuple[int, int]], bits: np.ndarray
) -> np.ndarray:
    """Sum subword surprisals into the word whose characters they cover."""
    spans = _word_starts(text)
    totals = np.zeros(len(spans), dtype=np.float64)
    edges = [s for s, _ in spans]
    for (start, end), value in zip(offsets, bits):
        # This tokenizer folds the preceding space into the token, so the raw
        # offset points at the end of the previous word. Skip past the space
        # before deciding which word the subword belongs to.
        while start < end and text[start].isspace():
            start += 1
        if start >= end:
            continue
        # a subword belongs to the last word that starts at or before it
        idx = int(np.searchsorted(edges, start, side="right")) - 1
        if 0 <= idx < len(totals):
            totals[idx] += value
    return totals


# --------------------------------------------------------------------------
# collecting windows
# --------------------------------------------------------------------------


def clean_position(alignment: Sequence[Optional[int]], boundary: int) -> int:
    """Where the boundary falls once the injected material is taken out."""
    return sum(1 for src in alignment[:boundary] if src is not None)


def collect_windows(
    rows: Sequence[dict], scorer: Surprisal, batch: int, verbose: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Gather ±WINDOW surprisal profiles for each disfluency type.

    Returns per-type arrays of shape (instances, 2*WINDOW+1) and, keyed the same
    way, the control window taken from the same sentence with the disfluency
    removed. Row i of a type's control array belongs to row i of its data, so
    the two can be differenced instance by instance.
    """
    wanted: List[dict] = []
    for row in rows:
        result = InjectionResult.from_dict(row)
        if not result.annotations:
            continue
        clean_spoken = [spoken_form(t) for t in result.clean_tokens]
        for ann in result.annotations:
            if ann.type not in TYPES:
                continue
            boundary = ann.boundary_index()
            if boundary is None:
                continue
            if boundary - WINDOW < 0 or boundary + WINDOW >= len(result.disfluent_tokens):
                continue
            centre = clean_position(result.alignment, boundary)
            if centre - WINDOW < 0 or centre + WINDOW >= len(clean_spoken):
                continue
            wanted.append(
                {
                    "type": ann.type,
                    "disfluent": " ".join(result.disfluent_tokens),
                    "clean": " ".join(clean_spoken),
                    "boundary": boundary,
                    "centre": centre,
                }
            )

    if verbose:
        print(f"{len(wanted)} disfluencies with a full window", file=sys.stderr)
    if not wanted:
        return {}, {}

    # Score each distinct sentence once.
    texts: List[str] = []
    index: Dict[str, int] = {}
    for item in wanted:
        for key in ("disfluent", "clean"):
            if item[key] not in index:
                index[item[key]] = len(texts)
                texts.append(item[key])
    if verbose:
        print(f"scoring {len(texts)} sentences", file=sys.stderr)

    scores: List[np.ndarray] = []
    for start in range(0, len(texts), batch):
        scores.extend(scorer.word_surprisals(texts[start : start + batch], batch))
        if verbose:
            print(f"  {min(start + batch, len(texts))}/{len(texts)}", end="\r",
                  file=sys.stderr)
    if verbose:
        print(file=sys.stderr)

    per_type: Dict[str, List[np.ndarray]] = defaultdict(list)
    per_control: Dict[str, List[np.ndarray]] = defaultdict(list)
    for item in wanted:
        dis = scores[index[item["disfluent"]]]
        cln = scores[index[item["clean"]]]
        b, c = item["boundary"], item["centre"]
        if b + WINDOW >= len(dis) or c + WINDOW >= len(cln):
            continue  # truncated by the LM's context window
        per_type[item["type"]].append(dis[b - WINDOW : b + WINDOW + 1])
        per_control[item["type"]].append(cln[c - WINDOW : c + WINDOW + 1])

    data = {k: np.stack(v) for k, v in per_type.items() if v}
    control = {k: np.stack(per_control[k]) for k in data}
    return data, control


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def mend_sign(value: float) -> str:
    """Format an effect size so the direction is never ambiguous."""
    return f"{value:+.2f}"


def bootstrap_ci(
    data: np.ndarray, n_draws: int = 2000, seed: int = 13
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and percentile 95% CI over instances, per relative position."""
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    means = data.mean(axis=0)
    draws = np.empty((n_draws, data.shape[1]))
    for i in range(n_draws):
        idx = rng.integers(0, n, n)
        draws[i] = data[idx].mean(axis=0)
    lo = np.percentile(draws, 2.5, axis=0)
    hi = np.percentile(draws, 97.5, axis=0)
    return means, lo, hi


def paired_difference_ci(
    data: np.ndarray, control: np.ndarray, n_draws: int = 2000, seed: int = 13
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CI on the per-instance difference, which is paired by construction."""
    diff = data - control
    return bootstrap_ci(diff, n_draws, seed)


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------


def make_figure(
    per_type: Dict[str, np.ndarray],
    per_control: Dict[str, np.ndarray],
    out_dir: Path,
    n_draws: int = 2000,
    tag: str = "",
    title: str = "",
) -> Tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "figure.dpi": 300,
        }
    )

    positions = np.arange(-WINDOW, WINDOW + 1)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    for name in TYPES:
        if name not in per_type:
            continue
        mean, lo, hi = bootstrap_ci(per_type[name], n_draws)
        ax.fill_between(positions, lo, hi, color=COLOURS[name], alpha=0.18, linewidth=0)
        ax.plot(
            positions,
            mean,
            color=COLOURS[name],
            linewidth=1.4,
            marker="o",
            markersize=2.8,
            label=f"{LABELS[name]} (n={per_type[name].shape[0]})",
        )

    pooled = np.concatenate([per_control[k] for k in TYPES if k in per_control])
    if pooled.size:
        mean, lo, hi = bootstrap_ci(pooled, n_draws)
        ax.fill_between(positions, lo, hi, color=BASELINE_COLOUR, alpha=0.12, linewidth=0)
        ax.plot(
            positions,
            mean,
            color=BASELINE_COLOUR,
            linewidth=1.2,
            linestyle="--",
            dashes=(4, 2),
            label=f"Fluent control (n={pooled.shape[0]})",
        )

    ax.axvline(0, color="#999999", linewidth=0.6, zorder=0)
    ax.set_xlabel("Position relative to repair boundary")
    ax.set_ylabel("Surprisal (bits)")
    ax.set_xticks(positions[::2])
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(y=0.16)
    ax.legend(frameon=False, loc="upper right", handlelength=1.6, borderaxespad=0.2)
    if title:
        ax.set_title(title, pad=4)
    fig.tight_layout(pad=0.4)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"surprisal_at_repair{tag}.pdf"
    png = out_dir / f"surprisal_at_repair{tag}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Surprisal around repair boundaries")
    ap.add_argument("--data", type=Path, default=Path("data/test.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--lm", default=DEFAULT_LM)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="cap the test sentences read")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    ap.add_argument("--title", default="", help="figure title")
    args = ap.parse_args(argv)

    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]

    scorer = Surprisal(args.lm, args.device)
    print(f"surprisal model: {args.lm} on {scorer.device}", file=sys.stderr)
    per_type, per_control = collect_windows(rows, scorer, args.batch)
    if not per_type:
        print("no disfluencies with a full window; nothing to plot", file=sys.stderr)
        return 1

    pdf, png = make_figure(
        per_type, per_control, args.out, args.draws, args.tag, args.title
    )

    # Effect size: the largest gap between a disfluency profile and its own
    # fluent control, differenced instance by instance and reported wherever
    # the peak falls, whichever direction that turns out to be.
    report: Dict[str, dict] = {}
    lines = []
    for name in TYPES:
        if name not in per_type:
            continue
        data, ctrl = per_type[name], per_control[name]
        mean_diff, lo, hi = paired_difference_ci(data, ctrl, args.draws)
        peak = int(np.argmax(np.abs(mean_diff)))
        rel = peak - WINDOW
        report[name] = {
            "n": int(data.shape[0]),
            "peak_position": rel,
            "peak_effect_bits": float(mean_diff[peak]),
            "peak_effect_ci95": [float(lo[peak]), float(hi[peak])],
            "surprisal_at_peak_bits": float(data.mean(axis=0)[peak]),
            "control_at_peak_bits": float(ctrl.mean(axis=0)[peak]),
            "effect_at_boundary_bits": float(mean_diff[WINDOW]),
            "effect_at_boundary_ci95": [float(lo[WINDOW]), float(hi[WINDOW])],
        }
        lines.append(
            f"{LABELS[name]:<13s} n={data.shape[0]:<5d} peak {mend_sign(mean_diff[peak])} "
            f"bits at position {rel:+d} "
            f"[95% CI {lo[peak]:+.2f}, {hi[peak]:+.2f}]; at the boundary itself "
            f"{mend_sign(mean_diff[WINDOW])} bits "
            f"[95% CI {lo[WINDOW]:+.2f}, {hi[WINDOW]:+.2f}]"
        )

    print("\nSurprisal relative to the matched fluent control:")
    for line in lines:
        print("  " + line)
    print(
        "\n  Each figure is a per-instance paired difference against that "
        "disfluency's\n  own control window. The dashed line in the figure pools "
        "the control\n  windows of all three types, so reading a gap straight off "
        "the plot will\n  not reproduce these numbers exactly."
    )
    print(f"\nwrote {pdf} and {png}")

    (args.out / f"surprisal_effects{args.tag}.json").write_text(
        json.dumps(report, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
