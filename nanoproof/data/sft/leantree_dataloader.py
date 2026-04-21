"""Tokenize ``(state, tactic, proof_depth)`` triples into batched
``(inputs, targets)`` tensors. Each input triple emits **two** training
samples: a tactic-prediction sample and a value-prediction sample.

Two entry points:

- ``rl_data_generator(generator, batch_size, ...)`` is the workhorse: takes
  any iterator of triples and yields ``(inputs, targets)`` batches forever.
- ``sft_data_generator(dataset, batch_size, ...)`` wraps a flat list with
  DDP striding + epoch progress tracking, then delegates to
  ``rl_data_generator`` and attaches ``(approx_progress, last_step)`` to each
  yielded batch. Used by the SFT training loop.
"""

import torch
from itertools import islice

from nanoproof.common import get_dist_info, GLOBAL_CONFIG
from nanoproof.tokenizer import get_tokenizer, value_to_token_ids
from nanoproof.data.sft.leantree import leantree_transitions


def rl_data_generator(generator, batch_size, device="cuda"):
    """Tokenize an iterator of ``(state, tactic, proof_depth)`` triples (or
    ``(state, tactic, proof_depth, source)`` 4-tuples) into batched
    ``(inputs, targets, sources)`` tensors. Each triple becomes two rows in
    the batch (one tactic-prediction sample, one value-prediction sample), so
    ``batch_size`` must be even.

    Triples whose tactic exceeds ``tactic_max_len`` or whose state+tactic
    exceeds ``max_seq_len`` are silently dropped. Stops when the upstream
    generator is exhausted.

    The ``sources`` element is a length-``batch_size`` list of the optional
    4th tuple field (``None`` if the upstream yielded 3-tuples). Both the
    tactic and value rows derived from the same triple share its source.
    """
    assert batch_size % 2 == 0, "batch_size must be even (each triple emits 2 samples)"
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    eos_token = tokenizer.get_eos_token_id()
    assert bos_token is not None
    assert eos_token is not None
    pad_token_id = tokenizer.encode_special("<|pad|>")
    tactic_delim_tok = tokenizer.encode_special("<|tactic|>")
    value_delim_tok = tokenizer.encode_special("<|value|>")

    def collate(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, _ in batch) - 1  # seq of n creates inputs/targets of n-1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)  # -1 is ignore index
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n - 1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            # mask[1:] omits the mask for the BOS token, which is never a target atm so it's ok
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1  # mask out targets where mask is 0
            targets[i, :n - 1] = row_targets
        return inputs.to(device), targets.to(device)

    batch = []
    batch_sources: list = []
    for item in generator:
        state, tactic, proof_depth = item[0], item[1], item[2]
        source = item[3] if len(item) > 3 else None
        state, tactic = state.strip(), tactic.strip()
        assert len(state) != 0 and len(tactic) != 0 and proof_depth >= 1

        state_toks = tokenizer.encode(state + "\n", prepend=bos_token)
        tactic_toks = tokenizer.encode(tactic, append=eos_token)
        proof_depth = min(proof_depth, GLOBAL_CONFIG.num_value_bins)
        value_toks = value_to_token_ids(tokenizer, proof_depth) + [eos_token]

        # these filtered triples are <0.1% of mathlib
        if len(tactic_toks) > GLOBAL_CONFIG.tactic_max_len:
            continue
        if len(state_toks) + 1 + len(tactic_toks) > GLOBAL_CONFIG.max_seq_len:
            continue
        assert len(state_toks) + 1 + len(value_toks) <= GLOBAL_CONFIG.max_seq_len

        batch.append((
            state_toks + [tactic_delim_tok] + tactic_toks,
            [0] * (len(state_toks) + 1) + [1] * len(tactic_toks),
        ))
        batch_sources.append(source)
        batch.append((
            state_toks + [value_delim_tok] + value_toks,
            [0] * (len(state_toks) + 1) + [1] * len(value_toks),
        ))
        batch_sources.append(source)

        if len(batch) == batch_size:
            inputs, targets = collate(batch)
            yield inputs, targets, list(batch_sources)
            batch = []
            batch_sources = []


def sft_data_generator(dataset, batch_size, device="cuda"):
    """SFT wrapper around ``rl_data_generator``: stream a flat list with DDP
    striding for an unbounded number of epochs, and tag each yielded batch
    with the current ``approx_progress`` (in [0, 1] within the current epoch)
    and a sticky ``last_step`` flag (set once the epoch has finished consuming
    its triples).
    """
    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    progress = {"approx": 0.0, "last_step": False}

    def stream_triples():
        while True:
            for i in range(ddp_rank, len(dataset), ddp_world_size):
                progress["approx"] = i / len(dataset)
                if i + ddp_world_size >= len(dataset):
                    progress["last_step"] = True
                yield dataset[i]
            print(f"Warning: Rank {ddp_rank} will loop again on leantree ({len(dataset)=}).", flush=True)

    for inputs, targets, _sources in rl_data_generator(stream_triples(), batch_size, device):
        yield inputs, targets, progress["approx"], progress["last_step"]


# -----------------------------------------------------------------------------
# CLI: inspect tokenized samples

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect SFT/RL tokenized batches from leantree", allow_abbrev=False)
    parser.add_argument("--split", choices=["train", "valid"], default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=10)
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = list(leantree_transitions(split=args.split))
    tokenizer = get_tokenizer()
    for inputs, targets, approx_progress, last_step in islice(
        sft_data_generator(dataset, batch_size=args.batch_size, device="cpu"), args.num_batches
    ):
        for i in range(inputs.size(0)):
            print(f"Input {i}:")
            print(inputs[i])
            print(tokenizer.decode(inputs[i].tolist()))
            print()
            print(f"Target {i}:")
            print(targets[i])
            # replace -1 with a different token so that it can be decoded and displayed
            targets[i][targets[i] == -1] = tokenizer.encode("X")[0]
            print(tokenizer.decode(targets[i].tolist()))
            print("--")
        print(f"approx_progress={approx_progress:.4f} last_step={last_step}")
        print("-" * 100)
