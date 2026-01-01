from dataclasses import dataclass
import math
import re
from typing import List, Literal, Optional, Tuple

from diffusers import ZImagePipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
import einops
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

# ----- VAE -----


@dataclass
class VAEConfig:
    block_out_channels: Tuple[int] = (128, 256, 512, 512)
    in_channels: int = 3
    latent_channels: int = 16
    out_channels: int = 3
    num_res_down_layers: int = 2
    num_res_up_layers: int = 3
    num_heads: int = 1
    norm_num_groups: int = 32
    dropout: float = 0.0
    eps: float = 1e-6
    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159

    def __post_init__(self):
        self.block_channels = self.block_out_channels[0]
        self.hidden_size = self.block_out_channels[-1]
        assert self.hidden_size % self.num_heads == 0


class VAEAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        num_heads = self.config.num_heads
        x = einops.rearrange(x, "B C H W -> B (H W) C")
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = einops.rearrange(q, "B L (head d_k) -> B head L d_k", head=num_heads)
        k = einops.rearrange(k, "B L (head d_k) -> B head L d_k", head=num_heads)
        v = einops.rearrange(v, "B L (head d_k) -> B head L d_k", head=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(x, "B head L d_k -> B L (head d_k)")
        x = self.o_proj(x)
        x = einops.rearrange(x, "B (H W) C -> B C H W", H=H, W=W)
        return x


class ResNet(nn.Module):
    def __init__(self, config, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(config.norm_num_groups, in_channels, config.eps)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(config.norm_num_groups, out_channels, config.eps)
        self.dropout = nn.Dropout(config.dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(x)
        else:
            residual = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = residual + x
        return x


class DownSampler(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 2, 0)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1))
        x = self.conv(x)
        return x


class UpSampler(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        config: VAEConfig,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        self.resnets = ResNet(config, in_channels, out_channels)
        resnets = [ResNet(config, in_channels, out_channels)]
        for i in range(1, num_layers):
            resnets.append(ResNet(config, out_channels, out_channels))
        self.resnets = nn.Sequential(*resnets)
        self.downsampler = None
        self.upsampler = None
        if downsample:
            self.downsampler = DownSampler(out_channels, out_channels)
        elif upsample:
            self.upsampler = UpSampler(out_channels, out_channels)

    def forward(self, x):
        x = self.resnets(x)
        if self.downsampler:
            x = self.downsampler(x)
        elif self.upsampler:
            x = self.upsampler(x)
        return x


class MidBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config.block_out_channels[-1]
        self.resnet1 = ResNet(config, channels, channels)
        self.norm_attn = nn.GroupNorm(config.norm_num_groups, channels, config.eps)
        self.attn = VAEAttention(config)
        self.resnet2 = ResNet(config, channels, channels)

    def forward(self, x):
        x = self.resnet1(x)
        x = x + self.attn(self.norm_attn(x))
        x = self.resnet2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config: VAEConfig = None):
        super().__init__()
        config = config or VAEConfig()
        self.config = config
        self.conv_in = nn.Conv2d(
            config.in_channels, config.block_out_channels[0], 3, 1, 1
        )
        down_blocks = []
        out_channels = config.block_out_channels[0]
        for i in range(len(config.block_out_channels)):
            is_final_block = i == len(config.block_out_channels) - 1
            in_channels = out_channels
            out_channels = config.block_out_channels[i]
            down_blocks.append(
                ResBlock(
                    config,
                    in_channels,
                    out_channels,
                    config.num_res_down_layers,
                    downsample=not is_final_block,
                )
            )
        self.down_blocks = nn.Sequential(*down_blocks)
        self.mid_block = MidBlock(config)
        self.norm = nn.GroupNorm(
            config.norm_num_groups, config.block_out_channels[-1], config.eps
        )
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            config.block_out_channels[-1], config.latent_channels * 2, 3, 1, 1
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down_blocks(x)
        x = self.mid_block(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv_out(x)
        return x

    def load(self, encoder):
        # Retrieve the state dictionary from the reference encoder
        state_dict = encoder.state_dict()
        new_state_dict = {}

        # Iterate over each key-value pair in the reference state dict
        for k, v in state_dict.items():
            # 1. Base key replacements: normalize naming conventions for norm and attention layers
            k = (
                # Replace final norm layer name: conv_norm_out -> norm
                k.replace("conv_norm_out", "norm")
                # Map attention projection layer names to target naming (to_q/k/v -> q/k/v_proj; to_out.0 -> o_proj)
                .replace("to_q", "q_proj")
                .replace("to_k", "k_proj")
                .replace("to_v", "v_proj")
                .replace("to_out.0", "o_proj")
            )

            # 2. Fix downsampler key mapping: convert plural "downsamplers" to singular "downsampler"
            # Exact match for downsamplers.0.conv -> downsampler.conv (core downsampling layer)
            if "downsamplers.0.conv" in k:
                k = k.replace("downsamplers.0.conv", "downsampler.conv")
            # Fallback for any remaining downsamplers keys (remove pluralization)
            elif "downsamplers" in k:
                k = k.replace("downsamplers", "downsampler")

            # 3. Fix mid_block key mappings for residual blocks and attention layers
            if "mid_block" in k:
                # Map mid_block residual blocks: resnets.0 -> resnet1, resnets.1 -> resnet2
                if "mid_block.resnets.0" in k:
                    k = k.replace("mid_block.resnets.0", "mid_block.resnet1")
                elif "mid_block.resnets.1" in k:
                    k = k.replace("mid_block.resnets.1", "mid_block.resnet2")

                # Map attention group norm to target norm_attn (mid_block.attentions.0.group_norm -> mid_block.norm_attn)
                elif "mid_block.attentions.0.group_norm" in k:
                    k = k.replace(
                        "mid_block.attentions.0.group_norm",
                        "mid_block.norm_attn",
                    )

                # Map attention layer path: mid_block.attentions.0 -> mid_block.attn
                elif "mid_block.attentions.0" in k:
                    k = k.replace("mid_block.attentions.0", "mid_block.attn")

            # Add the remapped key-value pair to the new state dictionary
            new_state_dict[k] = v

        # Load the remapped state dictionary with strict mode enabled
        # Strict mode ensures all keys match exactly (catches missing/unexpected keys)
        self.load_state_dict(new_state_dict, strict=True)


class Decoder(nn.Module):
    def __init__(self, config: VAEConfig = None):
        super().__init__()
        config = config or VAEConfig()
        self.config = config
        self.conv_in = nn.Conv2d(
            config.latent_channels, config.block_out_channels[-1], 3, 1, 1
        )
        self.mid_block = MidBlock(config)
        up_blocks = []
        out_channels = config.block_out_channels[-1]
        for i in reversed(range(len(config.block_out_channels))):
            is_first_block = i == 0
            is_final_block = i == len(config.block_out_channels) - 1
            in_channels = out_channels
            out_channels = config.block_out_channels[i]
            up_blocks.append(
                ResBlock(
                    config,
                    in_channels,
                    out_channels,
                    config.num_res_up_layers,
                    upsample=not is_first_block,
                )
            )
        self.up_blocks = nn.Sequential(*up_blocks)
        self.norm = nn.GroupNorm(
            config.norm_num_groups, config.block_out_channels[0], config.eps
        )
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            config.block_out_channels[0], config.out_channels, 3, 1, 1
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        x = self.up_blocks(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv_out(x)
        return x

    def load(self, decoder):
        # Retrieve the state dictionary from the reference decoder model
        state_dict = decoder.state_dict()
        new_state_dict = {}

        # Iterate through each key-value pair in the reference state dict for remapping
        for k, v in state_dict.items():
            # 1. Base key replacements: normalize core naming conventions
            k = (
                # Replace final norm layer name (conv_norm_out -> norm) to match target schema
                k.replace("conv_norm_out", "norm")
                # Map attention projection layer names to target q/k/v/o_proj naming
                .replace("to_q", "q_proj")
                .replace("to_k", "k_proj")
                .replace("to_v", "v_proj")
                .replace("to_out.0", "o_proj")
            )

            # 2. Fix upsampler key mapping: convert plural "upsamplers" to singular "upsampler"
            # Exact match for upsamplers.0.conv -> upsampler.conv (core upsampling layer)
            if "upsamplers.0.conv" in k:
                k = k.replace("upsamplers.0.conv", "upsampler.conv")
            # Fallback for any remaining upsamplers keys (remove pluralization)
            elif "upsamplers" in k:
                k = k.replace("upsamplers", "upsampler")

            # 3. Fix mid_block key mappings for residual blocks and attention layers
            if "mid_block" in k:
                # Map mid_block residual blocks: resnets.0 -> resnet1, resnets.1 -> resnet2
                if "mid_block.resnets.0" in k:
                    k = k.replace("mid_block.resnets.0", "mid_block.resnet1")
                elif "mid_block.resnets.1" in k:
                    k = k.replace("mid_block.resnets.1", "mid_block.resnet2")

                # Map attention group norm to target norm_attn (unify attention normalization naming)
                elif "mid_block.attentions.0.group_norm" in k:
                    k = k.replace(
                        "mid_block.attentions.0.group_norm",
                        "mid_block.norm_attn",
                    )

                # Map attention layer path: attentions.0 -> attn (simplify attention module naming)
                elif "mid_block.attentions.0" in k:
                    k = k.replace("mid_block.attentions.0", "mid_block.attn")

            # Add the remapped key-value pair to the new state dictionary
            new_state_dict[k] = v

        # Load the remapped state dictionary with strict mode enabled
        # Strict mode ensures full key matching (catches missing/unexpected keys)
        self.load_state_dict(new_state_dict, strict=True)


class VAE(nn.Module):
    def __init__(self, config: VAEConfig = None, device=None, dtype=None):
        super().__init__()
        config = config or VAEConfig()
        dd = {"device": device, "dtype": dtype}
        self.config = config
        self.encoder = Encoder(config).to(**dd)
        self.decoder = Decoder(config).to(**dd)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def load(self, vae):
        self.encoder.load(vae.encoder)
        self.decoder.load(vae.decoder)


# ----- text encoder -----


def rotate_half(x, interleave=False):
    if interleave:
        x1 = x[..., ::2, None]
        x2 = x[..., 1::2, None]
        return torch.cat((-x2, x1), dim=-1).reshape(x.shape)
    else:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, pos_embed, q_rotate=None, k_rotate=None):
    if q_rotate is None:
        q_rotate = rotate_half(q)
    if k_rotate is None:
        k_rotate = rotate_half(k)
    cos, sin = pos_embed
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (q_rotate * sin)
    k_embed = (k * cos) + (k_rotate * sin)
    return q_embed, k_embed


@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    padding_idx: Optional[int] = None
    hidden_size: int = 2560
    head_dim: Optional[int] = 128
    intermediate_size: int = 9728
    num_layers: int = 36
    num_attn_heads: int = 32
    num_kv_heads: int = 8
    theta: float = 1000000.0
    eps: float = 1e-6

    def __post_init__(self):
        assert self.hidden_size % self.num_attn_heads == 0
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attn_heads
        assert self.head_dim % 2 == 0
        assert self.num_attn_heads % self.num_kv_heads == 0
        self.num_kv_groups = self.num_attn_heads // self.num_kv_heads


class RoPE(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        half_dim = config.head_dim // 2
        indexes = torch.arange(half_dim, dtype=torch.float32)
        log_freq = -indexes * math.log(config.theta) / half_dim
        self.freq = torch.exp(log_freq)

    @torch.no_grad()
    def forward(self, x, position_ids):
        device = x.device
        dtype = x.dtype
        with torch.autocast(device_type=str(device), enabled=False):
            freq = self.freq.to(device=device, dtype=torch.float32)
            position_ids = position_ids.to(device, dtype=torch.float32)
            freq_pos = freq[None, None, :] * position_ids[:, :, None]
            cos = torch.cos(freq_pos)
            sin = torch.sin(freq_pos)
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)
        return cos.to(dtype), sin.to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        var = torch.mean(x**2, -1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x.to(dtype) * self.weight
        return x

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config):
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
        self.o_proj = nn.Linear(
            config.num_attn_heads * config.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = RMSNorm(config.head_dim, config.eps)
        self.k_norm = RMSNorm(config.head_dim, config.eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        pos_embed: Tuple[torch.Tensor, torch.Tensor],
    ):
        num_attn_heads = self.config.num_attn_heads
        num_kv_heads = self.config.num_kv_heads
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = einops.rearrange(q, "B L (head d_h) -> B L head d_h", head=num_attn_heads)
        k = einops.rearrange(k, "B L (head d_h) -> B L head d_h", head=num_kv_heads)
        v = einops.rearrange(v, "B L (head d_h) -> B head L d_h", head=num_kv_heads)
        q = einops.rearrange(self.q_norm(q), "B L head d_h -> B head L d_h")
        k = einops.rearrange(self.k_norm(k), "B L head d_h -> B head L d_h")
        q, k = apply_rope(q, k, pos_embed)
        if attention_mask is not None:
            attention_mask = attention_mask.to(bool)
        x = F.scaled_dot_product_attention(
            q, k, v, attention_mask, enable_gqa=True, is_causal=attention_mask is None
        )
        x = einops.rearrange(x, "B head L d_h -> B L (head d_h)")
        x = self.o_proj(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_ff = config.intermediate_size
        self.proj_gate = nn.Linear(config.hidden_size, d_ff, bias=False)
        self.proj_up = nn.Linear(config.hidden_size, d_ff, bias=False)
        self.act = nn.SiLU()
        self.proj_down = nn.Linear(d_ff, config.hidden_size, bias=False)

    def forward(self, x):
        x_gate = self.proj_gate(x)
        x_up = self.proj_up(x)
        x = self.act(x_gate) * x_up
        x = self.proj_down(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.attn = Qwen3Attention(config)
        self.norm_attn = RMSNorm(config.hidden_size, config.eps)
        self.ff = FeedForward(config)
        self.norm_ff = RMSNorm(config.hidden_size, config.eps)

    def forward(self, x, attention_mask=None, pos_embed=None):
        x = x + self.attn(self.norm_attn(x), attention_mask, pos_embed)
        x = x + self.ff(self.norm_ff(x))
        return x


class Qwen3(nn.Module):
    def __init__(self, config: Qwen3Config = None, device=None, dtype=None):
        super().__init__()
        config = config or Qwen3Config()
        dd = {"device": device, "dtype": dtype}
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.padding_idx, **dd
        )
        self.pos_embedding = RoPE(config).to(**dd)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config).to(**dd) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.eps).to(**dd)

    def forward(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids)
        pos_ids = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        pos_embed = self.pos_embedding(x, pos_ids)
        hidden_states = []
        for i, block in enumerate(self.blocks):
            x = block(x, attention_mask=attention_mask, pos_embed=pos_embed)
            hidden_states.append(x)
        x = self.norm(x)
        return x, hidden_states

    def load(self, model):
        # Retrieve the state dictionary from the reference model
        state_dict = model.state_dict()
        new_state_dict = {}

        # Step 1: Iterate through all key-value pairs and remap keys (no concatenation logic)
        for k, v in state_dict.items():
            # 1. Core key remapping: align naming conventions between reference and target model
            k_remapped = (
                k.replace(
                    "embed_tokens", "token_embedding"
                )  # Map token embedding layer name
                .replace(
                    "layers", "blocks"
                )  # Map transformer layer hierarchy (layers → blocks)
                .replace(
                    "self_attn", "attn"
                )  # Map self-attention module name (self_attn → attn)
                .replace("mlp", "ff")  # Map feed-forward network module name (mlp → ff)
                .replace(
                    "input_layernorm", "norm_attn"
                )  # Map input normalization layer
                .replace("down_proj", "proj_down")  # Map FFN down projection layer name
                .replace(
                    "post_attention_layernorm", "norm_ff"
                )  # Map post-attention normalization layer
                # Critical: Map original gate_proj/up_proj to new proj_gate/proj_up naming
                .replace("gate_proj", "proj_gate")
                .replace("up_proj", "proj_up")
            )

            # 2. Directly add all remapped keys to new_state_dict (no temporary caching)
            # Keep proj_gate and proj_up as separate entries (no concatenation)
            new_state_dict[k_remapped] = v

        # Step 2: Load the remapped state dictionary with strict mode (ensures key consistency)
        self.load_state_dict(new_state_dict, strict=True)


# ----- transformer -----

SEQ_MULTI_OF = 32


def split_and_pad(tensor, split_size, dim=0):
    tensors = tensor.split(split_size, dim=dim)
    return nn.utils.rnn.pad_sequence(tensors, batch_first=True)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


@dataclass
class TransformerConfig:
    latent_channels: int = 16
    patch_size: int = 2
    f_patch_size: int = 1
    hidden_size: int = 3840
    text_hidden_size: int = 2560
    num_heads: int = 30
    num_blocks: int = 30
    num_noise_refiners: int = 2
    num_text_refiners: int = 2
    adaln_emb_size: int = 256
    t_emb_interm_size: int = 1024
    t_scale: float = 1000.0
    max_period: float = 10000.0
    rope_theta: float = 256.0
    axes_dims: List[int] = (32, 48, 48)
    axes_lens: List[int] = (1536, 512, 512)
    out_channels: int = 3
    dropout: float = 0.0
    eps: float = 1e-5

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0
        assert self.hidden_size % 2 == 0
        self.flatten_channels = (
            self.latent_channels * self.f_patch_size * self.patch_size * self.patch_size
        )
        self.t_emb_size = min(self.hidden_size, self.adaln_emb_size)
        self.head_dim = self.hidden_size // self.num_heads
        assert len(self.axes_dims) == len(self.axes_lens)
        assert sum(self.axes_dims) == self.head_dim
        self.intermediate_size = int(self.hidden_size * 8 / 3)


class TimeEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.up = nn.Linear(config.t_emb_size, config.t_emb_interm_size)
        self.act = nn.SiLU()
        self.down = nn.Linear(config.t_emb_interm_size, config.t_emb_size)

        half_dim = config.t_emb_size // 2
        indexes = torch.arange(half_dim, dtype=torch.float32)
        log_freq = -math.log(config.max_period) * indexes / half_dim
        self.freq = torch.exp(log_freq)

    def forward(self, t):
        p = next(self.parameters())
        device = p.device
        dtype = p.dtype
        with torch.amp.autocast(str(device), enabled=False):
            if torch.is_tensor(t):
                t = t.to(device, torch.float32)
            else:
                t = torch.tensor(t, dtype=torch.float32, device=device)
            if t.ndim == 0:
                t = t[None]
            freq = self.freq.to(device=device, dtype=torch.float32)
            args = t[:, None] * freq[None, :]
            t_emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        t_emb = t_emb.to(device, dtype)
        t_emb = self.down(self.act(self.up(t_emb)))
        return t_emb


class RoPE3D(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.freqs = []
        for ax_len, ax_dim in zip(config.axes_lens, config.axes_dims):
            half_dim = ax_dim // 2
            indexes = torch.arange(half_dim, dtype=torch.float32)
            log_freq = -indexes * math.log(config.rope_theta) / half_dim
            self.freqs.append(torch.exp(log_freq))

    @torch.no_grad()
    def forward(self, x, position_ids):
        assert position_ids.shape[-1] == 3
        assert position_ids.ndim == 2
        device = x.device
        dtype = x.dtype
        coss = []
        sins = []
        with torch.autocast(device_type=str(device), enabled=False):
            position_ids = position_ids.to(device=device, dtype=torch.float32)
            for i, freq in enumerate(self.freqs):
                freq = freq.to(device, dtype=torch.float32)
                ids = position_ids[:, i]
                args = ids[:, None] * freq[None, :]
                cos = torch.cos(args)
                sin = torch.sin(args)
                # interleave to be consistent with ZImage implementation
                cos = cos.repeat_interleave(repeats=2, dim=-1)
                sin = sin.repeat_interleave(repeats=2, dim=-1)
                coss.append(cos.to(device, dtype))
                sins.append(sin.to(device, dtype))
        cos = torch.cat(coss, dim=-1)
        sin = torch.cat(sins, dim=-1)
        return cos, sin


class ZImageAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.eps)
        self.k_norm = RMSNorm(config.head_dim, config.eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_embed: Tuple[torch.Tensor, torch.Tensor],
    ):
        num_heads = self.config.num_heads
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = einops.rearrange(q, "B L (head d_h) -> B L head d_h", head=num_heads)
        k = einops.rearrange(k, "B L (head d_h) -> B L head d_h", head=num_heads)
        v = einops.rearrange(v, "B L (head d_h) -> B head L d_h", head=num_heads)
        q = einops.rearrange(self.q_norm(q), "B L head d_h -> B head L d_h")
        k = einops.rearrange(self.k_norm(k), "B L head d_h -> B head L d_h")

        q_split = torch.split(q, self.config.axes_dims, dim=-1)
        k_split = torch.split(k, self.config.axes_dims, dim=-1)
        q_rotate = torch.cat([rotate_half(_q_split, True) for _q_split in q_split], -1)
        k_rotate = torch.cat([rotate_half(_k_split, True) for _k_split in k_split], -1)
        q, k = apply_rope(q, k, pos_embed, q_rotate, k_rotate)

        attention_mask = attention_mask[:, None, None, :]
        x = F.scaled_dot_product_attention(q, k, v, attention_mask)
        x = einops.rearrange(x, "B head L d_h -> B L (head d_h)")
        x = self.o_proj(x)
        return x


class ZImageTransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, modulation=True):
        super().__init__()
        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Linear(config.t_emb_size, config.hidden_size * 4)

        self.norm_attn1 = RMSNorm(config.hidden_size, config.eps)
        self.attn = ZImageAttention(config)
        self.norm_attn2 = RMSNorm(config.hidden_size, config.eps)

        self.norm_ff1 = RMSNorm(config.hidden_size, config.eps)
        self.ff = FeedForward(config)
        self.norm_ff2 = RMSNorm(config.hidden_size, config.eps)

    def forward(self, x, attention_mask=None, pos_embed=None, c=None):
        if self.modulation:
            scale_attn, gate_attn, scale_ff, gate_ff = torch.chunk(
                self.adaLN_modulation(c).unsqueeze(1), 4, dim=-1
            )
            gate_attn = gate_attn.tanh()
            gate_ff = gate_ff.tanh()
            scale_attn = 1 + scale_attn
            scale_ff = 1 + scale_ff
        else:
            scale_attn, gate_attn, scale_ff, gate_ff = 1, 1, 1, 1
        x0 = x
        x = self.attn(self.norm_attn1(x) * scale_attn, attention_mask, pos_embed)
        x = x0 + gate_attn * self.norm_attn2(x)
        x0 = x
        x = self.ff(self.norm_ff1(x) * scale_ff)
        x = x0 + gate_ff * self.norm_ff2(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.t_emb_size, config.hidden_size)
        )
        self.norm_final = nn.LayerNorm(
            config.hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(config.hidden_size, config.flatten_channels)

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        x = self.linear(x)
        return x


class ZImageTransformer(nn.Module):
    def __init__(self, config: TransformerConfig = None, device=None, dtype=None):
        super().__init__()
        config = config or TransformerConfig()
        self.config = config
        dd = {"device": device, "dtype": dtype}
        self.time_embedding = TimeEmbedding(config).to(**dd)
        self.patch_embedding = nn.Linear(
            config.flatten_channels, config.hidden_size, **dd
        )
        self.text_embedding = nn.Sequential(
            RMSNorm(config.text_hidden_size, config.eps, **dd),
            nn.Linear(config.text_hidden_size, config.hidden_size, **dd),
        )
        self.pos_embedding = RoPE3D(config)
        self.latent_pad_token = nn.Parameter(torch.empty((1, config.hidden_size), **dd))
        self.text_pad_token = nn.Parameter(torch.empty((1, config.hidden_size), **dd))
        self.noise_refiners = nn.ModuleList(
            [
                ZImageTransformerBlock(config, modulation=True).to(**dd)
                for _ in range(config.num_noise_refiners)
            ]
        )
        self.text_refiners = nn.ModuleList(
            [
                ZImageTransformerBlock(config, modulation=False).to(**dd)
                for _ in range(config.num_text_refiners)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                ZImageTransformerBlock(config, modulation=True).to(**dd)
                for _ in range(config.num_blocks)
            ]
        )
        self.final_layer = FinalLayer(config).to(**dd)

    @torch.no_grad()
    def get_pos_id(self, shape, offset=(0, 0, 0), device=None, pad=0):
        seq_len = torch.prod(torch.tensor(shape))
        pos_id = torch.arange(seq_len, dtype=torch.int32, device=device)
        cols = torch.unravel_index(pos_id, shape)
        cols = [col + offset[i] for i, col in enumerate(cols)]
        pos_id = torch.stack(cols, dim=-1)
        pos_id_pad = torch.zeros((pad, 3), dtype=torch.int32, device=device)
        pos_id = torch.cat([pos_id, pos_id_pad], dim=0)
        return pos_id

    def patchify(self, latents: List[torch.Tensor], text_embeds: List[torch.Tensor]):
        pf = self.config.f_patch_size
        ph = self.config.patch_size
        pw = self.config.patch_size
        device = latents[0].device
        dd = {"device": device, "dtype": bool}
        latents_padded = []
        latent_masks = []
        latent_pos_ids = []
        latent_sizes = []
        text_embeds_padded = []
        text_masks = []
        text_pos_ids = []
        for i in range(len(latents)):
            latent = latents[i]
            text_embed = text_embeds[i]
            C, F, H, W = latent.shape
            L = text_embed.shape[0]
            latent_sizes.append((C, F, H, W))

            # ----- text -----
            text_pad = (-L) % SEQ_MULTI_OF
            mask = torch.cat([torch.zeros(L, **dd), torch.ones(text_pad, **dd)])
            text_masks.append(mask)
            text_embed = torch.cat(
                [text_embed, text_embed[-1:].expand(text_pad, -1)], dim=0
            )
            text_embeds_padded.append(text_embed)
            pos_id = self.get_pos_id((L + text_pad, 1, 1), (1, 0, 0), device=device)
            text_pos_ids.append(pos_id)

            # ----- latent -----
            f = F // pf
            h = H // ph
            w = W // pw
            latent = einops.rearrange(
                latent,
                "C (f pf) (h ph) (w pw) -> (f h w) (pf ph pw C)",
                pf=pf,
                ph=ph,
                pw=pw,
            )
            latent_pad = (-latent.shape[0]) % SEQ_MULTI_OF
            mask = torch.cat(
                [torch.zeros(latent.shape[0], **dd), torch.ones(latent_pad, **dd)]
            )
            latent_masks.append(mask)
            latent = torch.cat([latent, latent[-1:].expand(latent_pad, -1)], dim=0)
            latents_padded.append(latent)
            pos_id = self.get_pos_id(
                (f, h, w), (1 + L + text_pad, 0, 0), device=device, pad=latent_pad
            )
            latent_pos_ids.append(pos_id)

        return (
            latents_padded,
            latent_masks,
            latent_pos_ids,
            latent_sizes,
            text_embeds_padded,
            text_masks,
            text_pos_ids,
        )

    def unpatchify(self, latents: List[torch.Tensor], latent_sizes: List[Tuple[int]]):
        pf = self.config.f_patch_size
        ph = self.config.patch_size
        pw = self.config.patch_size
        new_latents = []
        for i in range(len(latents)):
            C, F, H, W = latent_sizes[i]
            f = F // pf
            h = H // ph
            w = W // pw
            latent = einops.rearrange(
                latents[i][: f * h * w],
                "(f h w) (pf ph pw C) -> C (f pf) (h ph) (w pw)",
                f=f,
                h=h,
                w=w,
                pf=pf,
                ph=ph,
                pw=pw,
            )
            new_latents.append(latent)
        return new_latents

    def prepare_hidden_states(
        self,
        tensors: List[torch.Tensor],
        masks: List[torch.Tensor],
        pos_ids: List[torch.Tensor],
        type: Literal["latent", "text", "unified"],
        cos=None,
        sin=None,
    ):
        assert type in ["latent", "text", "unified"]
        seq_lens = [tensor.shape[0] for tensor in tensors]
        tensors = torch.cat(tensors, dim=0)
        if type != "unified":
            masks = torch.cat(masks, dim=0)
            pos_ids = torch.cat(pos_ids, dim=0)

        if type == "latent":
            embeds = self.patch_embedding(tensors)
            embeds[masks] = self.latent_pad_token
        elif type == "text":
            embeds = self.text_embedding(tensors)
            embeds[masks] = self.text_pad_token
        else:
            embeds = tensors
        embeds = split_and_pad(embeds, seq_lens, dim=0)
        if cos is None:
            cos, sin = self.pos_embedding(embeds, pos_ids)
        cos = split_and_pad(cos, seq_lens, dim=0)
        sin = split_and_pad(sin, seq_lens, dim=0)
        attn_mask = torch.zeros(
            (len(seq_lens), max(seq_lens)), dtype=bool, device=embeds.device
        )
        for i, seq_len in enumerate(seq_lens):
            attn_mask[i, :seq_len] = True
        return seq_lens, embeds, attn_mask, cos, sin

    def forward(self, latents: List[torch.Tensor], t, text_embeds: List[torch.Tensor]):
        if not isinstance(latents, list):
            latents = [latents]
        if not isinstance(text_embeds, list):
            text_embeds = [text_embeds]
        assert len(latents) == len(text_embeds)
        B = len(latents)
        t_embeds = self.time_embedding(t * self.config.t_scale)

        (
            latents_padded,
            latent_masks,
            latent_pos_ids,
            latent_sizes,
            text_embeds_padded,
            text_masks,
            text_pos_ids,
        ) = self.patchify(latents, text_embeds)

        # ----- latent -----
        latent_seq_lens, latent_embeds, latent_attn_mask, latent_cos, latent_sin = (
            self.prepare_hidden_states(
                latents_padded, latent_masks, latent_pos_ids, type="latent"
            )
        )

        latent_hidden_states = latent_embeds
        for noise_refiner in self.noise_refiners:
            latent_hidden_states = noise_refiner(
                latent_hidden_states,
                latent_attn_mask,
                (latent_cos, latent_sin),
                t_embeds,
            )

        # ----- text -----
        text_seq_lens, text_embeds, text_attn_mask, text_cos, text_sin = (
            self.prepare_hidden_states(
                text_embeds_padded, text_masks, text_pos_ids, type="text"
            )
        )

        text_hidden_states = text_embeds
        for context_refiner in self.text_refiners:
            text_hidden_states = context_refiner(
                text_hidden_states, text_attn_mask, (text_cos, text_sin)
            )

        # ----- unified -----
        def get_unified(latent_tensor, text_tensor, index):
            l = latent_tensor[index][: latent_seq_lens[index]]
            t = text_tensor[index][: text_seq_lens[index]]
            return torch.cat([l, t], dim=0)

        hidden_states = []
        cos = []
        sin = []

        for i in range(B):
            hidden_states.append(
                get_unified(latent_hidden_states, text_hidden_states, i)
            )
            cos.append(get_unified(latent_cos, text_cos, i))
            sin.append(get_unified(latent_sin, text_sin, i))
        cos = torch.cat(cos, dim=0)
        sin = torch.cat(sin, dim=0)
        seq_lens, embeds, attn_mask, cos, sin = self.prepare_hidden_states(
            hidden_states, None, None, type="unified", cos=cos, sin=sin
        )

        hidden_states = embeds
        for block in self.blocks:
            hidden_states = block(hidden_states, attn_mask, (cos, sin), t_embeds)
        hidden_states = self.final_layer(hidden_states, t_embeds)

        latents = self.unpatchify(hidden_states, latent_sizes)
        return latents

    def load(self, model):
        """
        Load and remap reference model state dict to target model, compatible with full layer count (0-29)
        Fixes missing keys error by retaining all layers/blocks and using non-strict loading
        """
        # Step 1: Extract raw state dict from reference model
        ref_state_dict = model.state_dict()
        new_state_dict = {}

        # Step 2: Define core key remapping rules (covers all naming discrepancies)
        # Format: (old_substring_from_ref, new_substring_for_target)
        key_remap_rules = [
            # 1. Pad token naming alignment
            ("x_pad_token", "latent_pad_token"),
            ("cap_pad_token", "text_pad_token"),
            # 2. Time embedding layer mapping (t_embedder → time_embedding)
            ("t_embedder.mlp.0", "time_embedding.up"),
            ("t_embedder.mlp.2", "time_embedding.down"),
            # 3. Text embedding layer mapping (cap_embedder → text_embedding)
            ("cap_embedder", "text_embedding"),
            # 4. Refiner module naming (singular → plural + context → text)
            ("noise_refiner", "noise_refiners"),
            ("context_refiner", "text_refiners"),
            # 5. Attention module key mapping (attention → attn, to_q → q_proj, etc.)
            ("attention.norm_q", "attn.q_norm"),
            ("attention.norm_k", "attn.k_norm"),
            ("attention.to_q", "attn.q_proj"),
            ("attention.to_k", "attn.k_proj"),
            ("attention.to_v", "attn.v_proj"),
            ("attention.to_out.0", "attn.o_proj"),  # Flatten to_out.0 → o_proj
            # 6. FFN (Feed-Forward Network) mapping (w1/w2/w3 → proj_gate/proj_down/proj_up)
            ("feed_forward.w1", "ff.proj_gate"),
            ("feed_forward.w2", "ff.proj_down"),
            ("feed_forward.w3", "ff.proj_up"),
            # 7. Normalization layer naming alignment
            ("attention_norm1", "norm_attn1"),
            ("ffn_norm1", "norm_ff1"),
            ("attention_norm2", "norm_attn2"),
            ("ffn_norm2", "norm_ff2"),
            # 8. adaLN modulation layer mapping (ref's .0 → target's base name)
            ("adaLN_modulation.0", "adaLN_modulation"),
            # 9. Core transformer layer hierarchy (layers → blocks)
            ("layers", "blocks"),
            # 10. Final layer mapping (ref's all_final_layer.2-1 → target's final_layer)
            ("all_final_layer.2-1.", "final_layer."),
            ("all_x_embedder.2-1", "patch_embedding"),  # Patch embedding alignment
        ]

        # Step 3: Iterate through reference state dict and apply remapping
        for ref_key, ref_value in ref_state_dict.items():
            # Initialize target key with reference key
            target_key = ref_key

            # Apply all remapping rules sequentially
            for old_substr, new_substr in key_remap_rules:
                target_key = target_key.replace(old_substr, new_substr)

            # Special case: final_layer adaLN_modulation (ref .0 → target .1)
            if (
                "final_layer.adaLN_modulation.weight" in target_key
                or "final_layer.adaLN_modulation.bias" in target_key
            ):
                target_key = target_key.replace(
                    "adaLN_modulation.", "adaLN_modulation.1."
                )

            # Step 4: Updated Layer Filtering (RETAIN ALL VALID LAYERS)
            # Only skip invalid layers (not blocks 0-29, noise_refiners 0-1, text_refiners 0-1)
            skip_key = False

            # Keep noise_refiners 0-1 (skip any higher indices if exist)
            if "noise_refiners." in target_key:
                # Extract layer index (e.g., "noise_refiners.2.xxx" → 2)
                import re

                idx_match = re.search(r"noise_refiners\.(\d+)", target_key)
                if idx_match:
                    idx = int(idx_match.group(1))
                    if idx not in [0, 1]:
                        skip_key = True

            # Keep text_refiners 0-1 (skip any higher indices if exist)
            elif "text_refiners." in target_key:
                import re

                idx_match = re.search(r"text_refiners\.(\d+)", target_key)
                if idx_match:
                    idx = int(idx_match.group(1))
                    if idx not in [0, 1]:
                        skip_key = True

            # KEEP ALL blocks 0-29 (remove old filter that skipped blocks 1-29)
            # No filtering for blocks - retain all layers from reference model

            # Add valid remapped keys to new state dict
            if not skip_key:
                new_state_dict[target_key] = ref_value

        self.load_state_dict(new_state_dict)


# ----- zimage -----


class ZImage(nn.Module):
    def __init__(
        self,
        tokenizer: Qwen2Tokenizer,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae_config: VAEConfig = None,
        text_encoder_config: Qwen3Config = None,
        transformer_config: TransformerConfig = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.vae = VAE(vae_config, **dd)
        self.text_encoder = Qwen3(text_encoder_config, **dd)
        self.transformer = ZImageTransformer(transformer_config, **dd)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.max_text_len = self.transformer.config.axes_lens[0]

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def encode_text(self, text):
        encoding = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_text_len,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        embs = self.text_encoder(**encoding)[1][-2]
        embs = embs.to(self.device, self.dtype)
        return embs

    def encode_prompt(self, prompts, negative_prompts=None):
        if negative_prompts is None:
            negative_prompts = ""
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * len(prompts)
        assert len(prompts) == len(negative_prompts)

        for i in range(len(prompts)):
            prompts[i] = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompts[i]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            negative_prompts[i] = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": negative_prompts[i]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )

        prompt_embeds = self.encode_text(prompts)
        negative_prompt_embeds = self.encode_text(negative_prompts)
        return prompt_embeds, negative_prompt_embeds

    @torch.inference_mode()
    def forward(
        self,
        prompts,
        height=1024,
        width=1024,
        num_steps=50,
        negative_prompts=None,
        guidance_scale=5,
    ):
        device = self.device
        dtype = self.dtype

        # ----- embed prompts -----
        prompt_embeds, neg_prompt_embeds = self.encode_prompt(prompts, negative_prompts)
        batch_size = prompt_embeds.shape[0]
        prompt_embeds_model_input = list(prompt_embeds.unbind(dim=0)) + list(
            neg_prompt_embeds.unbind(dim=0)
        )

        # ----- sample latents -----
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        latent_channels = self.vae.config.latent_channels
        latent_shape = (batch_size, latent_channels, 1, latent_height, latent_width)
        patch_size = self.transformer.config.patch_size
        image_seq_len = (latent_height // patch_size) * (latent_width // patch_size)
        latents = torch.randn(latent_shape, device=device, dtype=dtype)

        # ----- prepare time steps -----
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        self.scheduler.set_timesteps(num_steps, device=device, mu=mu)
        ts = self.scheduler.timesteps

        # ----- denoise -----
        for i, t in enumerate(tqdm(ts)):
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(batch_size)
            timestep = (1000 - timestep) / 1000
            # Normalized time for time-aware config (0 at start, 1 at end)
            t_norm = timestep[0].item()

            cfg_truncation = 1.0
            current_guidance_scale = guidance_scale
            if t_norm > cfg_truncation:
                current_guidance_scale = 0.0

            latent_model_input = latents.to(self.dtype).repeat(2, 1, 1, 1, 1)
            latent_model_input_list = list(latent_model_input.unbind(dim=0))
            timestep_model_input = timestep.repeat(2)

            model_out_list = self.transformer(
                latent_model_input_list,
                timestep_model_input,
                prompt_embeds_model_input,
            )

            # Perform CFG
            pos_out = model_out_list[:batch_size]
            neg_out = model_out_list[batch_size:]

            noise_pred = []
            for j in range(batch_size):
                pos = pos_out[j].float()
                neg = neg_out[j].float()
                pred = pos + current_guidance_scale * (pos - neg)
                noise_pred.append(pred)

            noise_pred = torch.stack(noise_pred, dim=0)
            noise_pred = -noise_pred

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred.to(torch.float32), t, latents, return_dict=False
            )[0]
            assert latents.dtype == torch.float32

        latents = latents.squeeze(2).to(self.dtype)
        images = self.vae.decode(
            latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        )
        images = self.image_processor.postprocess(
            images, output_type="pil", do_denormalize=[True] * batch_size
        )
        return images

    def load(self, pipeline: ZImagePipeline):
        self.vae.load(pipeline.vae)
        self.text_encoder.load(pipeline.text_encoder)
        self.transformer.load(pipeline.transformer)
