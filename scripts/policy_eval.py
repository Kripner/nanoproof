import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_recall(model, leantree_batches, steps, token_bytes):
    for _ in range(steps):

