import torch
import sys
import os
from unittest.mock import MagicMock

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock nanoproof.common to avoid import errors and control get_dist_info
sys.modules["nanoproof.common"] = MagicMock()
# We also need to mock muon and adamw if we want to avoid importing them or if they have deps
# But let's try to only mock common first, or mock all if they are just utils
# Given model.py imports them, we can mock them to be safe and simple
sys.modules["nanoproof.muon"] = MagicMock()
sys.modules["nanoproof.adamw"] = MagicMock()

from nanoproof.model import Network, NetworkConfig, ValueHead

def test_network():
    print("Testing Network...")
    config = NetworkConfig(
        sequence_len=32,
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        num_value_bins=10
    )
    
    model = Network(config)
    model.init_weights()
    
    # Test forward pass
    bs, seq_len = 2, 16
    idx = torch.randint(0, config.vocab_size, (bs, seq_len))
    
    output = model(idx)
    
    print(f"Policy logits shape: {output.policy_logits.shape}")
    print(f"Value logits shape: {output.value_logits.shape}")
    
    assert output.policy_logits.shape == (bs, seq_len, config.vocab_size)
    assert output.value_logits.shape == (bs, seq_len, config.num_value_bins)
    
    # Test ValueHead to_scalar
    print("Testing ValueHead to_scalar...")
    value_head = model.value_head
    # Create logits that strongly favor the last bin (max value)
    logits = torch.zeros((bs, seq_len, config.num_value_bins))
    logits[..., -1] = 100.0
    
    scalar_val = value_head.to_scalar(logits)
    print(f"Scalar values shape: {scalar_val.shape}")
    print(f"Scalar values (should be close to max_value={config.max_value}): {scalar_val[0,0].item()}")
    
    assert scalar_val.shape == (bs, seq_len)
    assert torch.allclose(scalar_val, torch.tensor(config.max_value), atol=1e-3)
    
    # Test generate
    print("Testing generate...")
    tokens = [1, 2, 3]
    gen = model.generate(tokens, max_tokens=5)
    generated = list(gen)
    print(f"Generated tokens: {generated}")
    assert len(generated) == 5

    # Test setup_optimizers
    print("Testing setup_optimizers...")
    # Mock get_dist_info to return (ddp, rank, local_rank, world_size)
    # We mocked nanoproof.common, so we need to set the return value on that mock
    # Note: model.py imports get_dist_info FROM nanoproof.common.
    # Since we mocked sys.modules["nanoproof.common"] BEFORE import, 
    # the imported get_dist_info in model.py is the mock.
    # However, we need to access the SAME mock object to configure it.
    # We can access it via sys.modules["nanoproof.common"].get_dist_info
    sys.modules["nanoproof.common"].get_dist_info.return_value = (False, 0, 0, 1)
    
    optimizers = model.setup_optimizers()
    assert len(optimizers) == 2
    adamw_opt, muon_opt = optimizers
    
    # Check AdamW params
    # AdamW has 3 groups: lm_head, value_head, embedding
    assert len(adamw_opt.param_groups) == 3
    
    # Verify value_head params are in AdamW
    value_head_params = list(model.value_head.parameters())
    found_value_head = False
    for group in adamw_opt.param_groups:
        if any(p is value_head_params[0] for p in group['params']):
            found_value_head = True
            break
    assert found_value_head, "ValueHead params not found in AdamW optimizer"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_network()
