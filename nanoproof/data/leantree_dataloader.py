import torch
from itertools import islice

from nanoproof.common import get_dist_info
from nanoproof.tokenizer import get_tokenizer, value_to_token_ids
from nanoproof.data.leantree import iter_data
from nanoproof.model import NetworkConfig

STATE_MAX_LEN = 640
TACTIC_MAX_LEN = 128

def sft_data_generator(dataset, batch_size, device="cuda"):
    assert batch_size % 2 == 0  # need this because we generate both tactic and value samples for each datapoint
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    eos_token = tokenizer.get_eos_token_id()
    assert bos_token is not None
    assert eos_token is not None
    pad_token_id = tokenizer.encode_special("<|pad|>")
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, _ in batch) - 1  # seq of n creates inputs/targets of n-1
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
    last_step = False
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            state, tactic, proof_depth = dataset[i]
            state, tactic = state.strip(), tactic.strip()
            assert len(state) != 0 and len(tactic) != 0 and proof_depth >= 1

            state_toks = tokenizer.encode(state + "\n", prepend=bos_token)

            tactic_delim_tok = tokenizer.encode_special("<|tactic|>")
            tactic_toks = tokenizer.encode(tactic, append=eos_token)

            value_delim_tok = tokenizer.encode_special("<|value|>")
            value_toks = value_to_token_ids(tokenizer, proof_depth) + [eos_token]

            # these are <0.1% of mathlib
            if len(tactic_toks) > TACTIC_MAX_LEN:
                continue
            if len(state_toks) + 1 + len(tactic_toks) > 768:
                continue
            assert len(state_toks) + 1 + len(value_toks) <= 768

            batch.append((
                state_toks + [tactic_delim_tok] + tactic_toks,
                [0] * (len(state_toks) + 1) + [1] * len(tactic_toks)
            ))
            # TODO: uncomment this once we are using <|value|>
            # TODO: we also need to change the dataset size calculation in SFT.py accordingly!
            # TODO: we also need to change the tactic_eval script to distinguish between tactic and value samples
            # batch.append((
            #     state_toks + [value_delim_tok] + value_toks,
            #     [0] * (len(state_toks) + 1) + [1] * len(value_toks)
            # ))

            approx_progress = i / len(dataset)
            last_step = last_step or (i + ddp_world_size >= len(dataset))
            if len(batch) == batch_size:
                yield *collate_and_yield(batch), approx_progress, last_step
                batch = []
        print(f"Warning: Rank {ddp_rank} will loop again on leantree ({len(dataset)=}).", flush=True)

def rl_data_generator(generator, batch_size, device="cuda"):
    assert batch_size % 2 == 0  # need this because we generate both tactic and value samples for each datapoint
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    eos_token = tokenizer.get_eos_token_id()
    assert bos_token is not None
    assert eos_token is not None
    pad_token_id = tokenizer.encode_special("<|pad|>")

    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, _ in batch) - 1  # seq of n creates inputs/targets of n-1
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
    last_step = False
    for state, tactic, proof_depth in generator:
        state, tactic = state.strip(), tactic.strip()
        assert len(state) != 0 and len(tactic) != 0
        # assert proof_depth >= 1

        state_toks = tokenizer.encode(state + "\n", prepend=bos_token)

        tactic_delim_tok = tokenizer.encode_special("<|tactic|>")
        tactic_toks = tokenizer.encode(tactic, append=eos_token)

        # value_delim_tok = tokenizer.encode_special("<|value|>")
        # value_toks = value_to_token_ids(tokenizer, proof_depth) + [eos_token]

        # these are <0.1% of mathlib
        if len(tactic_toks) > TACTIC_MAX_LEN:
            continue
        if len(state_toks) + 1 + len(tactic_toks) > 768:
            continue
        # assert len(state_toks) + 1 + len(value_toks) <= 768

        batch.append((
            state_toks + [tactic_delim_tok] + tactic_toks,
            [0] * (len(state_toks) + 1) + [1] * len(tactic_toks)
        ))
        # TODO: uncomment this once we are using <|value|>
        # TODO: we also need to change the dataset size calculation in SFT.py accordingly!
        # TODO: we also need to change the tactic_eval script to distinguish between tactic and value samples
        # batch.append((
        #     state_toks + [value_delim_tok] + value_toks,
        #     [0] * (len(state_toks) + 1) + [1] * len(value_toks)
        # ))

        if len(batch) == batch_size:
            yield collate_and_yield(batch)
            batch = []

if __name__ == "__main__":
    print("Loading dataset...")
    dataset = list(iter_data(split="train"))
    tokenizer = get_tokenizer()
    for inputs, targets in islice(sft_data_generator(dataset, batch_size=4), 10):
        for i in range(inputs.size(0)):
            print(f"Input {i}:")
            print(inputs[i])
            print(tokenizer.decode(inputs[i].tolist()))
            print()

            print(f"Target {i}:")
            print(targets[i])
            # replace -1 with a different token so that it can be decoded
            targets[i][targets[i] == -1] = tokenizer.encode("X")[0]
            print(tokenizer.decode(targets[i].tolist()))
            print("--")

        print("-" * 100)