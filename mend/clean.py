"""Run the trained model over text.

The rest of the repo evaluates the model on prepared datasets; this is the one
entry point that just cleans up whatever you give it.

    $ echo "um send the the report to john no wait sarah by friday" | mend-clean
    Send the report to Sarah by Friday.

    $ mend-clean "remind me to book the flights on uh tuesday"
    Remind me to book the flights on Tuesday.

Input is normalised to the shape the model was trained on -- lowercased, edge
punctuation stripped -- so pasting text that already has capitals and commas
works too. Pass --raw to feed it through untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from mend.injector import spoken_form


def normalise(line: str) -> str:
    """Put a line into the register the model was fine-tuned on."""
    words = [spoken_form(tok) for tok in line.split()]
    return " ".join(w for w in words if w)


def read_lines(args: argparse.Namespace) -> List[str]:
    if args.text:
        return [" ".join(args.text)]
    return [line.strip() for line in sys.stdin if line.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Clean dictated text: remove fillers, resolve repairs, "
        "restore punctuation"
    )
    ap.add_argument("text", nargs="*", help="text to clean; omit to read stdin")
    ap.add_argument("--model", type=Path, default=Path("outputs/model"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--raw", action="store_true", help="do not normalise the input")
    ap.add_argument("--show-input", dest="show_input", action="store_true")
    args = ap.parse_args(argv)

    lines = read_lines(args)
    if not lines:
        print("nothing to clean; pass text as arguments or on stdin", file=sys.stderr)
        return 1

    if not args.model.exists():
        print(
            f"no model at {args.model}. Build the data and train one first;\n"
            "it takes about an hour on a laptop:\n"
            "  python -m mend.generate_data --train 50000 --val 2000 --test 10000\n"
            "  python -m mend.train --batch 32 --epochs 2 --eval-steps 400 "
            "--real 25000 --out outputs/mixed\n"
            "  ln -sfn mixed/model outputs/model\n"
            "--real needs the SwDA parse; see 'Running the stages' in the README.",
            file=sys.stderr,
        )
        return 1

    # imported here so --help stays fast and does not need torch
    from mend.evaluate import load_model, run_model

    prepared = lines if args.raw else [normalise(line) for line in lines]
    model, tokenizer, device = load_model(args.model, args.device)
    cleaned_lines = run_model(
        model, tokenizer, device, prepared, args.batch, progress=len(prepared) > 200
    )
    for source, cleaned in zip(lines, cleaned_lines):
        if args.show_input:
            print(f"in : {source}")
            print(f"out: {cleaned}")
        else:
            print(cleaned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
