import torch

from nanoproof.common import get_dist_info
from nanoproof.tokenizer import get_tokenizer, value_to_token_ids
from nanoproof.data.leantree import iter_data
from nanoproof.model import NetworkConfig

STATE_MAX_LEN = 1536
TACTIC_MAX_LEN = 512

def sft_data_generator(dataset, batch_size, cfg: NetworkConfig, device="cuda"):
    assert batch_size % 2 == 0  # need this because we generate both tactic and value samples for each datapoint
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    eos_token = tokenizer.get_eos_token_id()
    assert bos_token is not None
    assert eos_token is not None
    # TODO: change to <|eos|>
    pad_token_id = tokenizer.encode_special("<|endoftext|>")  # use <|endoftext|> as the pad token is ok, these positions are masked in the loss
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, _ in batch) - 1  # seq of n creates inputs/targets of n-1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)  # -1 is ignore index
        for i, (ids, mask) in enumerate(batch):
            print(ids)
            print(tokenizer.decode(ids))
            print("---")
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n - 1] = ids_tensor[:-1]
            # recall -1 is the ignore index, so mask out targets where mask is 0
            row_targets = ids_tensor[1:]
            # mask[1:] omits the mask for the BOS token, which is never a target atm so it's ok
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1  # mask out targets where mask is 0
            targets[i, :n - 1] = row_targets
        inputs = inputs.to(device)  # move to device
        targets = targets.to(device)
        return inputs, targets

    # iterates over the dataset in epochs, tokenizes
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            state, tactic, proof_depth = dataset[i]

            state_toks = tokenizer.encode(state, prepend=bos_token)

            tactic_delim_toks = tokenizer.encode("\n<|tactic|> ", prepend=bos_token)
            tactic_toks = tokenizer.encode(tactic, append=eos_token)

            value_delim_toks = tokenizer.encode("\n<|value|> ", prepend=bos_token)
            value_toks = value_to_token_ids(tokenizer, proof_depth, cfg)

            # these are <0.1% of mathlib and prevent OOM
            if len(tactic_toks) > 256:
                continue
            if len(state_toks) + len(tactic_delim_toks) + len(tactic_toks) > 1024:
                continue
            assert len(state_toks) + len(value_delim_toks) + len(value_toks) <= 1024

            batch.append((
                state_toks + tactic_delim_toks + tactic_toks,
                [0] * (len(state_toks) + len(tactic_delim_toks)) + [1] * len(tactic_toks)
            ))
            batch.append((
                state_toks + value_delim_toks + value_toks,
                [0] * (len(state_toks) + len(value_delim_toks)) + [1] * len(value_toks)
            ))

            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []
