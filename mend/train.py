"""Fine-tune the cleanup model.

A seq2seq fine-tune of flan-t5-small:

    input   clean up: send it to john no wait sarah um by friday
    target  Send it to Sarah by Friday.

Usage::

    python -m mend.train --smoke      # 200 steps, checks the loss is moving
    python -m mend.train              # the real run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

PREFIX = "clean up: "
DEFAULT_MODEL = "google/flan-t5-small"


# --------------------------------------------------------------------------
# hardware
# --------------------------------------------------------------------------


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_pairs(path: Path, limit: int = 0) -> List[dict]:
    """Read only the two fields training needs.

    The JSONL also carries tokens, alignments and span annotations for the
    evaluation and surprisal stages. Keeping all of that for 50k rows costs
    hundreds of megabytes for no benefit here.
    """
    rows: List[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append({"disfluent": record["disfluent"], "clean": record["clean"]})
            if limit and len(rows) >= limit:
                break
    return rows


def to_dataset(rows: Sequence[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "disfluent": [r["disfluent"] for r in rows],
            "clean": [r["clean"] for r in rows],
        }
    )


def build_tokenize_fn(tokenizer, max_source: int, max_target: int):
    def fn(batch: Dict[str, List[str]]) -> Dict[str, List]:
        model_inputs = tokenizer(
            [PREFIX + t for t in batch["disfluent"]],
            max_length=max_source,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["clean"],
            max_length=max_target,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return fn


# --------------------------------------------------------------------------
# smoke check
# --------------------------------------------------------------------------


class LossTrace(TrainerCallback):
    """Keeps the training losses so the smoke run can check they fall."""

    def __init__(self) -> None:
        self.losses: List[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(float(logs["loss"]))


class TrimMPSCache(TrainerCallback):
    """Drop the MPS allocator's cached blocks every so often.

    The allocator keeps a separate pool per tensor shape and does not hand
    memory back. With dynamically padded batches the number of distinct shapes
    keeps growing, so the pool grows all run and eventually pushes the machine
    into swap, which shows up as step times that get steadily worse.
    """

    def __init__(self, every: int = 100) -> None:
        self.every = every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every == 0:
            torch.mps.empty_cache()


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------


@torch.no_grad()
def greedy_decode(
    model, tokenizer, texts: Sequence[str], device: str, max_new_tokens: int = 128
) -> List[str]:
    model.eval()
    out: List[str] = []
    for i in range(0, len(texts), 16):
        chunk = [PREFIX + t for t in texts[i : i + 16]]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=192).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=1,
                             do_sample=False)
        out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return out


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def train(args: argparse.Namespace) -> int:
    device = args.device or pick_device()
    print(f"device: {device}", file=sys.stderr)

    train_rows = load_pairs(args.data / "train.jsonl", args.max_train)
    val_rows = load_pairs(args.data / "val.jsonl")

    epochs = args.epochs
    if device == "cpu" and not args.smoke:
        # A full run on CPU is not worth the wall clock; shrink it and say so.
        train_rows = train_rows[:10_000]
        epochs = 1
        print(
            "no GPU: cutting the training set to 10000 pairs and 1 epoch",
            file=sys.stderr,
        )
    if args.smoke:
        train_rows = train_rows[: args.batch * args.smoke_steps]
        val_rows = val_rows[:200]
        epochs = 1

    print(f"train pairs: {len(train_rows)}  val pairs: {len(val_rows)}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q", "v"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    lr = args.lr if args.lr is not None else (3e-4 if args.lora else 5e-5)

    tokenize = build_tokenize_fn(tokenizer, args.max_source, args.max_target)
    ds_train = to_dataset(train_rows).map(
        tokenize, batched=True, remove_columns=["disfluent", "clean"]
    )
    ds_val = to_dataset(val_rows).map(
        tokenize, batched=True, remove_columns=["disfluent", "clean"]
    )

    out_dir = args.out / ("smoke" if args.smoke else "checkpoints")
    steps_per_epoch = max(1, math.ceil(len(ds_train) / args.batch))
    eval_every = args.eval_steps or max(50, min(500, steps_per_epoch // 2))

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        max_steps=args.smoke_steps if args.smoke else -1,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=lr,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=args.log_steps,
        eval_strategy="steps",
        eval_steps=eval_every,
        save_strategy="no" if args.smoke else "steps",
        save_steps=eval_every,
        save_total_limit=2,
        load_best_model_at_end=not args.smoke,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=(device == "cuda"),
        # Step time here is dominated by sequence length rather than batch
        # size, so batching similar lengths together is a large win.
        group_by_length=True,
        # MPS and CPU both want plain fp32 here; half precision on MPS is not
        # reliable for T5.
        fp16=(device == "cuda"),
        disable_tqdm=False,
    )

    trace = LossTrace()
    callbacks: List[TrainerCallback] = [trace]
    if device == "mps":
        callbacks.append(TrimMPSCache())
    if not args.smoke:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        # Quantising the padded width keeps the number of distinct tensor
        # shapes small, which the MPS allocator is much happier with.
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8),
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    resume = args.resume or None
    if resume == "auto":
        existing = sorted(
            out_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        resume = str(existing[-1]) if existing else None
        if resume:
            print(f"resuming from {resume}", file=sys.stderr)
    result = trainer.train(resume_from_checkpoint=resume)
    metrics = trainer.evaluate()
    final_eval_loss = metrics["eval_loss"]

    if args.smoke:
        losses = trace.losses
        if len(losses) < 2:
            print("smoke run produced too few log points to judge", file=sys.stderr)
            return 1
        head = sum(losses[:2]) / 2
        tail = sum(losses[-2:]) / 2
        print(f"\nsmoke: first logged loss {head:.4f} -> last {tail:.4f}")
        print(f"smoke: eval loss {final_eval_loss:.4f}")
        if tail >= head:
            print("smoke: loss did not fall, aborting", file=sys.stderr)
            return 1
        print("smoke: loss is falling")
        samples = [r["disfluent"] for r in val_rows[:5]]
        for src, hyp in zip(samples, greedy_decode(model, tokenizer, samples, device)):
            print(f"  in : {src}\n  out: {hyp}")
        return 0

    # Point outputs/model at the best checkpoint.
    best = trainer.state.best_model_checkpoint
    link = args.out / "model"
    if best:
        target = Path(best).resolve()
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
            else:
                import shutil

                shutil.rmtree(link)
        link.symlink_to(os.path.relpath(target, args.out))
        print(f"best checkpoint: {target}", file=sys.stderr)
    else:
        trainer.save_model(str(link))

    history = [h for h in trainer.state.log_history if "eval_loss" in h]
    first_eval = history[0]["eval_loss"] if history else float("nan")
    best_eval = min(h["eval_loss"] for h in history) if history else float("nan")
    summary = {
        "device": device,
        "model": args.model,
        "lora": bool(args.lora),
        "learning_rate": lr,
        "batch_size": args.batch,
        "epochs": epochs,
        "train_pairs": len(train_rows),
        "train_runtime_s": round(result.metrics.get("train_runtime", 0.0), 1),
        "first_eval_loss": first_eval,
        "best_eval_loss": best_eval,
        "final_train_loss": result.metrics.get("train_loss"),
    }
    (args.out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not math.isnan(first_eval) and best_eval >= first_eval:
        print("validation loss did not improve", file=sys.stderr)
        return 1

    print("\nten greedy decodes from the validation split:\n")
    samples = val_rows[:10]
    hyps = greedy_decode(model, tokenizer, [r["disfluent"] for r in samples], device)
    for row, hyp in zip(samples, hyps):
        print(f"  in  : {row['disfluent']}")
        print(f"  out : {hyp}")
        print(f"  gold: {row['clean']}\n")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune the Mend cleanup model")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lora", action="store_true", help="parameter-efficient tuning")
    ap.add_argument("--max-source", dest="max_source", type=int, default=128)
    ap.add_argument("--max-target", dest="max_target", type=int, default=128)
    ap.add_argument("--max-train", dest="max_train", type=int, default=0)
    ap.add_argument("--log-steps", dest="log_steps", type=int, default=20)
    ap.add_argument("--eval-steps", dest="eval_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--resume",
        default=None,
        help='checkpoint to carry on from, or "auto" for the latest one',
    )
    ap.add_argument("--smoke", action="store_true", help="short run to check the loss falls")
    ap.add_argument("--smoke-steps", dest="smoke_steps", type=int, default=200)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
