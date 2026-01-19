from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.models.qwen3_moe import Qwen3MoeModel


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, pos_embeddings: tuple[torch.Tensor]
) -> torch.Tensor:
    cos, sin = pos_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = cos * q + sin * rotate_half(q)
    k_embed = cos * k + sin * rotate_half(k)
    return q_embed, k_embed


@dataclass
class Qwen3MoeConfig:
    vocab_size: int = 151936
    padding_idx: Optional[int] = None
    hidden_size: int = 2048
    head_dim: Optional[int] = 128
    intermediate_size: int = 6144
    # moe_intermediate_size: int = 768
    moe_intermediate_size: int = 128
    # num_layers: int = 48
    num_layers: int = 1
    num_attn_heads: int = 32
    num_kv_heads: int = 4
    # num_experts: int = 128
    num_experts: int = 16
    num_experts_per_tok: int = 8
    theta: float = 1000000.0
    eps: float = 1e-6

    def __post_init__(self):
        assert self.hidden_size % self.num_attn_heads == 0
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attn_heads
        assert self.head_dim % 2 == 0
        assert self.num_attn_heads % self.num_kv_heads == 0
        self.num_kv_groups = self.num_attn_heads // self.num_kv_heads


class Rope(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.config = config
        half_dim = self.config.head_dim // 2
        index = torch.arange(half_dim, dtype=torch.float32)
        freqs = -index / half_dim * math.log(config.theta)
        self.freqs = torch.exp(freqs)
        self.attention_scaling = 1.0

    def forward(self, x, pos_ids):
        device = x.device
        dtype = x.dtype
        pos_ids = pos_ids.to(device)
        freqs = self.freqs.to(device)
        with torch.autocast(str(device), enabled=False):
            emb = pos_ids[None, :, None].float() * freqs[None, None, :].float()
            emb = torch.cat([emb, emb], dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype), sin.to(dtype)


class Attention(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attn_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * config.head_dim, bias=False
        )
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.eps)
        self.o_proj = nn.Linear(
            config.num_attn_heads * config.head_dim, config.hidden_size, bias=False
        )
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        pos_embeddings,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
    ) -> torch.Tensor:
        B, L = x.shape[:2]
        D = self.config.head_dim
        q = self.q_norm(self.q_proj(x).view(B, L, -1, D)).swapaxes(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, L, -1, D)).swapaxes(1, 2)
        v = self.v_proj(x).view(B, L, -1, D).swapaxes(1, 2)
        q, k = apply_rope(q, k, pos_embeddings)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        x = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=attention_mask is None,
            enable_gqa=True,
        )
        x = self.o_proj(x.swapaxes(1, 2).reshape(B, L, -1))
        return x


class MLP(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, is_expert=True):
        super().__init__()
        if is_expert:
            intermediate_size = config.moe_intermediate_size
        else:
            intermediate_size = config.intermediate_size
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act = nn.SiLU()
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return x


class SparseMLP(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x = x.view(B * L, D)
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.config.num_experts_per_tok, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)
        expert_mask = nn.functional.one_hot(
            selected_experts, self.config.num_experts
        ).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        output = torch.zeros_like(x)
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            topk_idx, token_idx = torch.where(expert_mask[expert_idx][0])
            expert_x = (
                expert_layer(x[token_idx]) * routing_weights[token_idx, topk_idx, None]
            )  # (len(token_idx), D)
            output.index_add_(0, token_idx, expert_x)
        x = output.view(B, L, D)
        return x


class Layer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm_attn = nn.RMSNorm(config.hidden_size, eps=config.eps)
        self.attn = Attention(config, layer_idx)
        self.norm_mlp = nn.RMSNorm(config.hidden_size, eps=config.eps)
        self.mlp = SparseMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        pos_embeddings,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm_attn(x), pos_embeddings, attention_mask, past_key_values
        )
        x = x + self.mlp(self.norm_mlp(x))
        return x


class Qwen3Moe(nn.Module):
    def __init__(self, config: Qwen3MoeConfig = None, device=None, dtype=None):
        super().__init__()
        config = config or Qwen3MoeConfig()
        self.config = config
        dd = {"device": device, "dtype": dtype}
        self.embed = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.padding_idx, **dd
        )
        self.rope = Rope(config)
        self.layers = nn.ModuleList(
            [Layer(config, layer_idx=i).to(**dd) for i in range(config.num_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.eps, **dd)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed(input_ids)
        if past_key_values is None:
            start = 0
        else:
            start = past_key_values.get_seq_length()
        pos_ids = torch.arange(
            start, hidden_states.shape[1], device=hidden_states.device
        )
        pos_embeddings = self.rope(hidden_states, pos_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                pos_embeddings=pos_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
        hidden_states = self.norm(hidden_states)
        logits = self.head(hidden_states)
        return logits

    def load(self, model):
        """
        Load reference model weights by remapping state dict keys using string replacement.
        Supports any number of transformer layers (layers.N) without regex dependency.

        Args:
            model: Reference model instance whose state dict needs to be remapped
        """
        state_dict = model.state_dict()
        new_state_dict = {}

        for k, v in state_dict.items():
            k_remapped = (
                k
                # Remove root "model." prefix first
                .replace("model.", "")
                # Map embedding layer name
                .replace("embed_tokens", "embed")
                # Map attention module name
                .replace("self_attn", "attn")
                # Map normalization layers
                .replace("input_layernorm", "norm_attn").replace(
                    "post_attention_layernorm", "norm_mlp"
                )
                # Map head layer name
                .replace("lm_head", "head")
            )

            new_state_dict[k_remapped] = v

        # Step 2: Load the remapped state dictionary with strict mode
        self.load_state_dict(new_state_dict, strict=True)
