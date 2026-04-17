# Source: https://github.com/karpathy/nanochat/blob/master/nanochat/dataloader.py
"""
BOS-aligned dataloader with Best-Fit Cropping over Nemotron parquet shards.

Every row starts with BOS. Documents are packed using a best-fit algorithm
yielding ~100% utilization (no padding) at the cost of cropping ~35% of tokens.
Approximate resume is supported via a per-batch state_dict.
"""

from collections import deque

import torch
import pyarrow.parquet as pq

from nanoproof.common import get_dist_info
from nanoproof.data.pretrain.nemotron import list_parquet_files
from nanoproof.tokenizer import get_tokenizer


def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.
    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch)).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    parquet_paths = list_parquet_files()
    assert len(parquet_paths) != 0, "No dataset parquet files found."
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def nemotron_batches_with_state(
    B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000,
):
    """BOS-aligned best-fit-cropping pretraining dataloader.

    Yields ``(inputs, targets, state_dict)`` triples. The state_dict can be
    threaded back via ``resume_state_dict`` to approximately resume training.

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly
    """
    assert split in ("train", "valid"), f"Invalid split: {split!r}"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict


def nemotron_batches(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in nemotron_batches_with_state(*args, **kwargs):
        yield inputs, targets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the Nemotron BOS-bestfit dataloader", allow_abbrev=False)
    parser.add_argument("--split", choices=["train", "valid"], default="train")
    parser.add_argument("-B", type=int, default=128, help="batch size")
    parser.add_argument("-T", type=int, default=1024, help="sequence length")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, default=10)
    args = parser.parse_args()

    dataloader = nemotron_batches_with_state(args.B, args.T, args.split, device=args.device)
    for i, (inputs, targets, state_dict) in enumerate(dataloader):
        if i >= args.max_batches:
            break
        print(f"Batch {i}: inputs={tuple(inputs.shape)} targets={tuple(targets.shape)} state={state_dict}", flush=True)
    print("Done.")