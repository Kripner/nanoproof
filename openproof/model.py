from collections import defaultdict
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core import Action


@dataclass
class ModelConfig:
    n_embd: int = 2048  # n_head * head_size = 16 * 128
    n_head: int = 16
    n_layer: int = 6
    n_encoder_layers: int = 18
    n_decoder_layers: int = 24
    head_size: int = 128
    ff_widening_factor: int = 6
    max_seq_len: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # value head
    num_value_bins: int = 64
    value_weight: float = 1e-3

    # TODO: sync with tokenizer
    vocab_size: int = 32000
    bos_token_id: int = 0
    eos_token_id: int = 1

    # sampling
    policy_num_tactics: int = 6
    max_tactic_len: int = 32


@dataclass
class NetworkTrainingOutput:
    """Output of the network during training."""
    value_logits: torch.Tensor
    policy_logits: torch.Tensor


@dataclass
class NetworkSamplingOutput:
    """Output of the network when sampling actions."""
    action_logprobs: dict[list[int], float]
    value: float


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.ff_widening_factor * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.ff_widening_factor * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class EncoderBlock(nn.Module):
    """Encoder block with self-attention and feed-forward layers."""

    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ff = MLP(config)
        self.norm1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x, src_mask=None, src_padding_mask=None):
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=src_mask,
            key_padding_mask=src_padding_mask
        )
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)

        return x


class DecoderBlock(nn.Module):
    """Decoder block with self-attention, cross-attention, and feed-forward layers."""

    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            batch_first=True
        )
        self.ff = MLP(config)
        self.norm1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

    def forward(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
    ):
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(attn_out)
        tgt = self.norm1(tgt)

        # Cross-attention with residual connection
        cross_attn_out, _ = self.cross_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(cross_attn_out)
        tgt = self.norm2(tgt)

        # Feed-forward with residual connection
        ff_out = self.ff(tgt)
        tgt = tgt + self.dropout3(ff_out)
        tgt = self.norm3(tgt)

        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(config) for _ in range(config.n_encoder_layers)
        ])

    def forward(self, x, src_mask=None, src_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, src_padding_mask=src_padding_mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(config) for _ in range(config.n_decoder_layers)
        ])

    def forward(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_padding_mask=None,
            memory_key_padding_mask=None,
    ):
        for layer in self.layers:
            tgt = layer(
                tgt, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        return tgt


class ValueHead(nn.Module):
    """Predicts of a state value as a categorical distribution."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.n_embd, config.num_value_bins, bias=False)

    def forward(self, encoded: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        """
        Forward pass of the value head.
        Args:
            encoded: Encoder output [batch_size, src_len, n_embd]
            src_key_padding_mask: Padding mask [batch_size, src_len] where True indicates padding
        """
        # Mean pooling over sequence length, ignoring padding
        if src_key_padding_mask is not None:
            # Invert mask: True for valid tokens, False for padding
            valid_mask = ~src_key_padding_mask  # [batch_size, src_len]
            # Sum over sequence, counting only valid tokens
            valid_counts = valid_mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]
            # Mask out padding positions
            masked_encoded = encoded * valid_mask.unsqueeze(-1).float()  # [batch_size, src_len, n_embd]
            # Mean pool
            pooled = masked_encoded.sum(dim=1) / valid_counts.clamp(min=1.0)  # [batch_size, n_embd]
        else:
            # Simple mean pooling if no mask
            pooled = encoded.mean(dim=1)  # [batch_size, n_embd]

        value_logits = self.proj(pooled)  # [batch_size, num_value_bins]
        return value_logits


class Network(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.n_embd)

        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.value_head = ValueHead(config)

        # untied weights
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all encoder blocks
        for block in self.encoder.layers:
            torch.nn.init.zeros_(block.ff.c_proj.weight)
        # zero out c_proj weights in all decoder blocks
        for block in self.decoder.layers:
            torch.nn.init.zeros_(block.ff.c_proj.weight)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd

        # Separate parameters into groups
        matrix_params = []
        for module in [self.encoder, self.decoder]:
            matrix_params.extend(list(module.parameters()))

        embedding_params = list(self.token_embedding.parameters()) + list(self.pos_embedding.parameters())
        lm_head_params = list(self.lm_head.parameters())
        value_head_params = list(self.value_head.parameters())

        # Scale learning rates based on model dimension
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Create parameter groups with appropriate learning rates
        param_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=value_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=matrix_params, lr=matrix_lr),
        ]

        # Create optimizer
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(param_groups, **adamw_kwargs)

        # Set initial learning rates
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

        return [optimizer]

    def forward(
            self,
            src_ids,
            tgt_ids,
            src_mask=None,
            tgt_mask=None,
            src_key_padding_mask=None,
            tgt_key_padding_mask=None,
            memory_mask=None,
            memory_key_padding_mask=None
    ) -> NetworkTrainingOutput:
        """
        Forward pass of the encoder-decoder Transformer.
        
        Args:
            src_ids: Source token ids [batch_size, src_len]
            tgt_ids: Target token ids [batch_size, tgt_len]
            src_mask: Source attention mask [src_len, src_len]
            tgt_mask: Target attention mask [tgt_len, tgt_len] (causal mask for autoregressive)
            src_key_padding_mask: Source padding mask [batch_size, src_len]
            tgt_key_padding_mask: Target padding mask [batch_size, tgt_len]
            memory_mask: Cross-attention mask [tgt_len, src_len]
            memory_key_padding_mask: Cross-attention padding mask [batch_size, src_len]
        """
        B, src_len = src_ids.shape
        _, tgt_len = tgt_ids.shape

        src_pos = torch.arange(0, src_len, device=src_ids.device).unsqueeze(0).expand(B, -1)
        src_emb = self.token_embedding(src_ids) + self.pos_embedding(src_pos)
        src_emb = self.dropout(src_emb)

        tgt_pos = torch.arange(0, tgt_len, device=tgt_ids.device).unsqueeze(0).expand(B, -1)
        tgt_emb = self.token_embedding(tgt_ids) + self.pos_embedding(tgt_pos)
        tgt_emb = self.dropout(tgt_emb)

        encoded = self.encoder(
            src_emb,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        decoder_out = self.decoder(
            tgt_emb,
            encoded,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        policy_logits = self.lm_head(decoder_out)  # [batch_size, tgt_len, vocab_size]

        value_logits = self.value_head(encoded, src_key_padding_mask=src_key_padding_mask)  # [batch_size, num_value_bins]

        return NetworkTrainingOutput(
            value_logits=value_logits,
            policy_logits=policy_logits
        )

    @torch.inference_mode()
    def sample(self, observation_ids) -> NetworkSamplingOutput:
        """
        Predict value and sample K tactics from a single observation.
        Inefficient - just for testing.
        """
        self.eval()
        src_len = observation_ids.shape[0]
        device = observation_ids.device
        K = self.config.policy_num_tactics

        src_pos = torch.arange(0, src_len, device=device).unsqueeze(0)
        src_emb = self.token_embedding(observation_ids) + self.pos_embedding(src_pos)
        src_emb = self.dropout(src_emb)
        encoded = self.encoder(src_emb)  # [1, src_len, n_embd]

        value_logits = self.value_head(encoded)  # [1, num_value_bins]
        value_probs = F.softmax(value_logits, dim=-1)  # [1, num_value_bins]

        bin_centers = torch.linspace(-1.0, 1.0, self.config.num_value_bins, device=device)
        value = (value_probs * bin_centers).sum(dim=-1).item()  # scalar

        ids = torch.full((K, 1), self.config.bos_token_id, dtype=torch.long, device=device)  # [K, 1]
        action_logprobs_list = [0.0] * K
        active_mask = torch.ones(K, dtype=torch.bool, device=device)
        lengths = torch.full((K,), self.config.max_tactic_len, dtype=torch.long, device=device)

        for t in range(self.config.max_tactic_len):
            if not active_mask.any():
                break

            network_output = self.forward(observation_ids, ids)

            logits = network_output.policy_logits[:, -1, :]  # [K, vocab_size]
            logprobs = F.log_softmax(logits, dim=-1)  # [K, vocab_size]
            probs = F.softmax(logits, dim=-1)  # [K, vocab_size]

            next_ids = torch.multinomial(probs, num_samples=1)  # [K, 1]
            eos_mask = (next_ids == 1).squeeze(-1)  # [K]

            for k in range(K):
                if active_mask[k]:
                    action_logprobs_list[k] += logprobs[k, next_ids[k, 0]].item()
                    if eos_mask[k]:
                        active_mask[k] = False
                        lengths[k] = t + 1

            ids = torch.cat([ids, next_ids], dim=1)  # [K, tgt_len+1]

        action_logprobs = {}
        for k in range(K):
            token_ids = ids[k, 1:lengths[k]].tolist()  # skip start token
            action_logprobs[tuple(token_ids)] = action_logprobs_list[k] / lengths[k]

        return NetworkSamplingOutput(
            action_logprobs=action_logprobs,
            value=value
        )
