import os

import torch
from leantree import LeanProject, LeanLibrary, LeanLibraries
from leantree.repl_adapter.server import LeanClient

from nanoproof.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type
from nanoproof.checkpoints import load_model, save_checkpoint
from nanoproof.engine import Engine
from nanoproof.data.minif2f import list_theorems, get_imports

"""
leanserver --project-path ~/troja/nanoproof/leantree_project/ --repl-exe ~/repos/leantree/lean-repl/.lake/build/bin/repl --imports Mathlib FormalConjectures.ForMathlib.Analysis.SpecialFunctions.NthRoot FormalConjectures.Util.Answer --max-processes 2 --address=10.10.24.9 --log-level=DEBUG
"""

source = "sft" # which checkpoint to load the model from
model_tag = "d20" # model tag to load the model from
device_type = "" # cuda|cpu|mps (empty => autodetect)
base_dir = get_base_dir()
server_address = "10.10.24.9"
server_port = 8000

device_type = autodetect_device_type() if device_type == "" else device_type
device = torch.device(device_type)

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
    branch = env.proof_from_sorry(theorem)
    print(branch.state())
    # zero, succ = branch.apply_tactic("cases n")