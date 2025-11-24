import torch

from nanoproof.common import get_dist_info
from nanoproof.tokenizer import get_tokenizer
from nanoproof.data.leantree import iter_data

STATE_MAX_LEN = 1536
TACTIC_MAX_LEN = 512

def sft_data_generator(batch_size, split, device="cuda"):
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    assert bos_token is not None
    # TODO: change to <|eos|>
    pad_token_id = tokenizer.encode_special("<|endoftext|>")  # use <|endoftext|> as the pad token is ok, these positions are masked in the loss
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    dataset = list(iter_data(split=split))

    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids in batch) - 1  # seq of n creates inputs/targets of n-1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)  # -1 is ignore index
        for i, (ids, mask) in enumerate(batch):
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
            state, tactic = dataset[i]
            state_tokens = tokenizer.encode(state, prepend=bos_token)
            tactic_tokens = tokenizer.encode("[TACTIC] " + tactic)

            token_lists = state_tokens + tactic_tokens
            mask = [0] * len(state_tokens) + [1] * len(tactic_tokens)
            batch.append((token_lists, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []
