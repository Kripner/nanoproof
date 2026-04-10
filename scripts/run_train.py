"""
End-to-end training pipeline runner.

Runs pretrain -> midtrain -> sft -> prover_eval (minif2f) -> rl in sequence,
chaining each stage's final checkpoint into the next. Each stage runs as its
own subprocess (and creates its own log/model dir). This script also creates a
top-level log dir under logs/train/ that contains a per-stage log file plus
symlinks to the per-stage log/model directories.

Usage:

    python scripts/run_train.py --run my_run

Run only some stages (skipped stages are simply not executed):

    python scripts/run_train.py --run my_run --stages sft,rl --start-model-path midtrain/.../model_001000.pt

Pass extra args to individual stages:

    python scripts/run_train.py --run my_run \\
        --pretrain-args="--depth=8 --num-iterations=1000" \\
        --rl-args="--lean-server=10.10.25.31:8000"

For now this keeps things simple: every stage is launched as a single-process
`python -m ...` (no torchrun), and RL runs in non-distributed mode.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from nanoproof.common import get_base_dir

ALL_STAGES = ["pretrain", "midtrain", "sft", "rl"]


def find_latest_checkpoint(model_dir: str) -> str:
    ckpts = list(Path(model_dir).glob("model_*.pt"))
    if not ckpts:
        raise RuntimeError(f"No model_*.pt checkpoints found in {model_dir}")
    latest = max(ckpts, key=lambda p: int(p.stem.removeprefix("model_")))
    return str(latest)


def run_subprocess(name: str, cmd: list[str], train_log_dir: Path) -> tuple[str | None, str | None]:
    """Run a stage subprocess, tee output to a per-stage log file, and parse out
    its log_dir / model_dir from the captured output (if reported)."""
    print(f"\n{'=' * 70}\n[run_train] >>> {name}\n{'=' * 70}", flush=True)
    print(f"[run_train] cmd: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)

    log_path = train_log_dir / f"{name}.log"
    log_dir: str | None = None
    model_dir: str | None = None

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            m = re.search(r"Log directory:\s*(\S+)", line)
            if m:
                log_dir = m.group(1)
            m = re.search(r"Model directory:\s*(\S+)", line)
            if m:
                model_dir = m.group(1)
        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(f"{name} failed with exit code {rc} (see {log_path})")

    return log_dir, model_dir


def link_into_train_dir(train_log_dir: Path, link_name: str, target: str) -> None:
    link = train_log_dir / link_name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def main():
    parser = argparse.ArgumentParser(description="End-to-end nanoproof training pipeline")
    parser.add_argument("--run", type=str, default="dummy", help="run name (used as wandb run name in each stage)")
    parser.add_argument("--stages", type=str, default=",".join(ALL_STAGES),
                        help=f"comma-separated subset of {ALL_STAGES} to run, in order")
    parser.add_argument("--start-model-path", type=str, default=None,
                        help="model path fed into the first stage; required if 'pretrain' is not in --stages")
    for st in ALL_STAGES:
        parser.add_argument(f"--{st}-args", type=str, default="",
                            help=f"extra CLI args forwarded to the {st} stage (single quoted string)")
    parser.add_argument("--prover-eval-args", type=str, default="",
                        help="extra CLI args forwarded to the post-sft prover_eval invocation")
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in ALL_STAGES:
            parser.error(f"unknown stage {s!r}; valid: {ALL_STAGES}")
    if "pretrain" not in stages and not args.start_model_path:
        parser.error("--start-model-path is required when 'pretrain' is not in --stages")

    extra = {st: shlex.split(getattr(args, f"{st}_args")) for st in ALL_STAGES}
    extra_prover = shlex.split(args.prover_eval_args)

    # Top-level train log dir
    base = get_base_dir()
    timestamp = datetime.now().strftime("%H-%M-%S_%d-%m-%y")
    train_log_dir = Path(base) / "logs" / "train" / f"{timestamp}_{args.run}"
    train_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run_train] Train log directory: {train_log_dir}")
    with open(train_log_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    current_model = args.start_model_path

    for stage in stages:
        if stage == "pretrain":
            cmd = [sys.executable, "-m", "nanoproof.pretrain", "--run", args.run] + extra["pretrain"]
        elif stage == "midtrain":
            cmd = [sys.executable, "-m", "nanoproof.midtrain",
                   "--run", args.run, "--model-path", current_model] + extra["midtrain"]
        elif stage == "sft":
            cmd = [sys.executable, "-m", "nanoproof.sft",
                   "--run", args.run, "--model-path", current_model] + extra["sft"]
        elif stage == "rl":
            # Non-distributed: empty --infra-file disables distributed mode in rl.py
            cmd = [sys.executable, "-m", "nanoproof.rl",
                   "--run", args.run, "--model-path", current_model,
                   "--infra-file", ""] + extra["rl"]
        else:
            raise AssertionError(stage)

        log_dir, model_dir = run_subprocess(stage, cmd, train_log_dir)
        if log_dir is None or model_dir is None:
            raise RuntimeError(f"{stage}: could not parse log/model directory from output")
        link_into_train_dir(train_log_dir, f"{stage}_log", log_dir)
        link_into_train_dir(train_log_dir, f"{stage}_model", model_dir)

        current_model = find_latest_checkpoint(model_dir)
        print(f"[run_train] {stage} done. Final checkpoint: {current_model}")

        # After SFT, run prover_eval on minif2f before continuing to RL.
        if stage == "sft":
            pe_cmd = [sys.executable, "scripts/prover_eval.py",
                      "--model-path", current_model,
                      "--datasets", "minif2f"] + extra_prover
            run_subprocess("prover_eval_minif2f", pe_cmd, train_log_dir)
            # prover_eval writes its results next to the sft checkpoint dir;
            # the per-stage .log file in train_log_dir captures stdout.

    print(f"\n[run_train] Pipeline complete. Logs: {train_log_dir}")


if __name__ == "__main__":
    main()
