import os

import torch
from leantree import LeanProject

from nanoproof.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type
from nanoproof.checkpoints import load_model, save_checkpoint
from nanoproof.engine import Engine
from nanoproof.data.leantree import iter_data
from nanoproof.data.leantree_dataloader import sft_data_generator
from nanoproof.data.minif2f import list_theorems, get_imports

source = "sft" # which checkpoint to load the model from
model_tag = "d20" # model tag to load the model from
device_type = "" # cuda|cpu|mps (empty => autodetect)
base_dir = get_base_dir()

device_type = autodetect_device_type() if device_type == "" else device_type
device = torch.device(device_type)

project_dir = os.path.join(base_dir, "leantree_project")
if not os.path.exists(project_dir) or not os.listdir(project_dir):
    project = LeanProject.create(project_dir)
else:
    project = LeanProject(project_dir)
# lean_imports = get_imports()
# print(lean_imports + "\n")

model, tokenizer, meta = load_model(source, device, phase="eval", model_tag=model_tag)
engine = Engine(model, tokenizer)

theorem = list_theorems("Valid")[0]
print(theorem + "\n-----")

with project.environment() as env:
    env.send_command("import Mathlib")
    branch = env.proof_from_sorry(theorem)
    # zero, succ = branch.apply_tactic("cases n")