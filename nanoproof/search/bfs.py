import os
from contextlib import nullcontext

import torch
from leantree import LeanProject, LeanLibrary, LeanLibraries
from leantree.repl_adapter.server import LeanClient

from nanoproof.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type
from nanoproof.checkpoints import load_model, save_checkpoint
from nanoproof.engine import Engine
from nanoproof.data.minif2f import list_theorems, get_imports

"""
leanserver --project-path ~/troja/nanoproof/leantree_project/ --repl-exe ~/repos/leantree/lean-repl/.lake/build/bin/repl --imports Mathlib FormalConjectures.ForMathlib.Analysis.SpecialFunctions.NthRoot FormalConjectures.Util.Answer --max-processes 2 --address=<PUBLIC_IP> --log-level=DEBUG
"""

source = "sft" # which checkpoint to load the model from
model_tag = "d26" # model tag to load the model from
device_type = "" # cuda|cpu|mps (empty => autodetect)
dtype = "bfloat16"
base_dir = get_base_dir()
server_address = "10.10.25.9"
server_port = 8000

device_type = autodetect_device_type() if device_type == "" else device_type
device = torch.device(device_type)
ptdtype = torch.float32 if dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

project_dir = os.path.join(base_dir, "leantree_project")
if not os.path.exists(project_dir) or not os.listdir(project_dir):
    # TODO: we need to add this to LeantreeProject.lean:
    # """
    # import FormalConjectures.ForMathlib.Analysis.SpecialFunctions.NthRoot
    # import FormalConjectures.Util.Answer
    # """
    formal_conjectures = LeanLibrary(
        name = "formal_conjectures",
        scope = "google-deepmind",
        git = "https://github.com/google-deepmind/formal-conjectures",
        rev = "d3d568c9b6ba0b0609b8dd61d0019cd77462e96a",
    )
    project = LeanProject.create(project_dir, libraries=[LeanLibraries.MATHLIB, formal_conjectures])
else:
    project = LeanProject(project_dir)

model, tokenizer, meta = load_model(source, device, phase="eval", model_tag=model_tag)
engine = Engine(model, tokenizer)
bos_token = tokenizer.get_bos_token_id()

theorem = list_theorems("Valid")[0]
print(theorem + "\n-----")

# We expect that the server has these imports:
# import Mathlib
# import FormalConjectures.ForMathlib.Analysis.SpecialFunctions.NthRoot
# import FormalConjectures.Util.Answer

client = LeanClient(server_address, server_port)
print(f"Connected to server at {server_address}:{server_port}")
print(f"Server status: {client.check_status()}")
with client.get_process() as env:
    print("Sending `open scoped` commands...")
    env.send_command("""
open scoped Real
open scoped Nat
open scoped Topology
open scoped Polynomial
""")
    print("Starting proof...")
    init_branch = env.proof_from_sorry(theorem)
    print(f"Initial state:\n{init_branch.state}")
    open_branches = [init_branch]
    proof = []
    rng = torch.Generator(device=device)
    rng.manual_seed(0)
    while open_branches:
        branch = open_branches.pop(0)
        state_str = str(branch.state).strip()
        print("-" * 80)
        print(f"Solving state:\n{state_str}\n")
        for retry_idx in range(10):
            print("Generating ..." + f" (retry {retry_idx})" if retry_idx != 0 else "")
            tokens = tokenizer(state_str + "\n<|tactic|>", prepend=bos_token)
            # print(" ".join([tokenizer.id_to_token(token) for token in tokens]))
            print(tokens)
            print(tokenizer.decode(tokens))
            with autocast_ctx:
                seed = torch.randint(torch.iinfo(torch.int32).max, (1,), device=device, generator=rng).item()
                sample_toks, masks = engine.generate_batch(tokens, num_samples=1, min_tokens=1, max_tokens=64, seed=seed)
            tactic_toks = [token for token, mask in zip(sample_toks[0], masks[0]) if mask == 1]
            tactic = tokenizer.decode(tactic_toks)
            print(f"Trying tactic:\n'{tactic}'")
            new_branches = branch.try_apply_tactic(tactic)
            if new_branches.is_success():
                proof.append(tactic)
                new_branches = new_branches.value
                print(f"Got {len(new_branches)} new branch(es)!")
                open_branches.extend(new_branches)
                break
            print(f"Error: '{new_branches.error}'\n")
        else:
            print("Could not generate a valid tactic, terminating.")
            break
    else:
        print(f"Proof found!\n--{"\n".join(proof)}\n--")