from dataclasses import dataclass
import math
import re
from typing import Tuple

from diffusers import StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
import einops
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers.activations import QuickGELUActivation
from transformers.models.clip import CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextTransformer


class Attention(nn.Module):
    def __init__(self, config, d_model=None, d_model_cross=None, bias=True):
        super().__init__()
        self.config = config
        self.is_causal = getattr(config, "is_causal", False)
        d_model = d_model or config.d_model
        d_model_cross = d_model_cross or d_model
        self.d_model = d_model
        self.d_model_cross = d_model_cross
        self.Wq = nn.Linear(d_model, d_model, bias)
        self.Wk = nn.Linear(d_model_cross, d_model, bias)
        self.Wv = nn.Linear(d_model_cross, d_model, bias)
        self.Wo = nn.Linear(d_model, d_model)
        if hasattr(config, "dropout"):
            self.dropout = nn.Dropout(config.dropout)
        else:
            self.dropout = None

    def forward(self, x, x_encoder=None):
        ndim = x.ndim
        if ndim == 4:
            height, width = x.shape[2], x.shape[3]
            x = einops.rearrange(x, "B C H W -> B (H W) C")
        if x_encoder is None:
            x_encoder = x
        elif x_encoder.ndim == 4:
            x_encoder = einops.rearrange(x_encoder, "B C H W -> B (H W) C")

        h = self.config.num_heads
        d_k = self.d_model // h

        Q = einops.rearrange(
            self.Wq(x), "B L (head d_k) -> B head L d_k", head=h, d_k=d_k
        )
        K = einops.rearrange(
            self.Wk(x_encoder), "B L (head d_k) -> B head L d_k", head=h, d_k=d_k
        )
        V = einops.rearrange(
            self.Wv(x_encoder), "B L (head d_k) -> B head L d_k", head=h, d_k=d_k
        )
        x = F.scaled_dot_product_attention(Q, K, V, is_causal=self.is_causal)
        x = einops.rearrange(x, "B head L d_k -> B L (head d_k)")
        x = self.Wo(x)

        if self.dropout is not None:
            x = self.dropout(x)
        if ndim == 4:
            x = einops.rearrange(x, "B (H W) C -> B C H W", H=height, W=width)

        return x


# ----- VAE -----


@dataclass
class VAEConfig:
    encoder_in_channels: int = 3
    latent_channels: int = 4
    block_out_channels: Tuple[int] = (128, 256, 512, 512)
    num_layers: int = 1
    down_layer_resnets: int = 2
    up_layer_resnets: int = 3
    num_heads: int = 1
    padding: int = 0
    norm_num_groups: int = 32
    eps: float = 1e-6
    dropout: float = 0.0
    scaling_factor: float = 0.18215

    def __post_init__(self):
        self.block_channels = self.block_out_channels[0]
        self.d_model = self.block_out_channels[-1]
        assert self.block_channels % self.norm_num_groups == 0
        assert self.d_model % self.num_heads == 0
        self.d_k = self.d_model // self.num_heads
        self.num_blocks = len(self.block_out_channels)
        self.encoder_out_channels = self.latent_channels * 2
        self.decoder_in_channels = self.latent_channels
        self.decoder_out_channels = self.encoder_in_channels


class ResNet(nn.Module):
    def __init__(self, config, in_channels, out_channels, emb_channels=None):
        super().__init__()
        self.config = config
        self.norm1 = nn.GroupNorm(config.norm_num_groups, in_channels, eps=config.eps)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.emb_proj = None
        if emb_channels is not None:
            self.emb_proj = nn.Linear(emb_channels, out_channels)
        self.norm2 = nn.GroupNorm(config.norm_num_groups, out_channels, eps=config.eps)
        self.dropout = nn.Dropout(config.dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward_proj(self, x, emb=None):
        x = self.act(x)
        x = self.conv1(x)
        if emb is not None and self.emb_proj is not None:
            emb = self.act(emb)
            emb = self.emb_proj(emb)
            emb = einops.rearrange(emb, "B C -> B C 1 1")
            x = x + emb
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

    def forward(self, x, emb=None):
        if self.conv_shortcut is not None:
            x0 = self.conv_shortcut(x)
        else:
            x0 = x
        return x0 + self.forward_proj(self.norm1(x), emb)


class AttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.resnet = ResNet(config, d_model, d_model)
        self.norm_attn = nn.GroupNorm(config.norm_num_groups, d_model, config.eps)
        self.attn = Attention(config)

    def forward(self, x):
        x = self.resnet(x)
        x = x + self.attn(self.norm_attn(x))
        return x


class DownSampler(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 2, padding)

    def forward(self, x):
        if self.padding == 0:
            x = F.pad(x, (0, 1, 0, 1))
        x = self.conv(x)
        return x


class UpSampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        config,
        in_channels,
        out_channels,
        num_layers,
        down_sampling=False,
        up_sampling=False,
        emb_channels=None,
    ):
        super().__init__()
        self.config = config
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                res_in_channels = in_channels
            else:
                res_in_channels = out_channels
            self.resnets.append(
                ResNet(config, res_in_channels, out_channels, emb_channels)
            )
        self.downsampler = None
        self.upsampler = None
        if down_sampling:
            assert not up_sampling
            self.downsampler = DownSampler(out_channels, out_channels, config.padding)
        elif up_sampling:
            self.upsampler = UpSampler(out_channels, out_channels)

    def forward(self, x, emb=None):
        for resnet in self.resnets:
            x = resnet(x, emb)
        if self.downsampler is not None:
            x = self.downsampler(x)
        elif self.upsampler is not None:
            x = self.upsampler(x)
        return x


class MidBlock(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.Sequential(
            *[AttentionBlock(config) for _ in range(config.num_layers)]
        )
        self.resnet = ResNet(config, config.d_model, config.d_model)

    def forward(self, x):
        x = self.blocks(x)
        x = self.resnet(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        self.conv_in = nn.Conv2d(
            config.encoder_in_channels, config.block_channels, 3, 1, 1
        )
        down_blocks = []
        out_channels = config.block_channels
        for i in range(config.num_blocks):
            in_channels = out_channels
            is_final_block = i == config.num_blocks - 1
            out_channels = config.block_out_channels[i]
            block = ResBlock(
                config,
                in_channels,
                out_channels,
                config.down_layer_resnets,
                down_sampling=not is_final_block,
            )
            down_blocks.append(block)
        self.down_blocks = nn.Sequential(*down_blocks)
        self.mid_block = MidBlock(config)
        self.norm = nn.GroupNorm(config.norm_num_groups, out_channels, eps=config.eps)
        self.act = nn.SiLU()
        out_channels = config.encoder_out_channels
        self.conv_out = nn.Conv2d(config.d_model, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down_blocks(x)
        x = self.mid_block(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv_out(x)
        return x

    def load(self, encoder):
        state_dict = encoder.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            # Base key replacements for attention layer parameter naming
            k = (
                k.replace("conv_norm_out", "norm")
                .replace("to_k", "Wk")
                .replace("to_q", "Wq")
                .replace("to_v", "Wv")
                .replace("to_out.0", "Wo")
            )

            # Fix downsampler key mapping: downsamplers.0.conv -> downsampler.conv
            if "downsamplers.0.conv" in k:
                k = k.replace("downsamplers.0.conv", "downsampler.conv")
            # Handle other downsamplers keys by removing pluralization
            elif "downsamplers" in k:
                k = k.replace("downsamplers", "downsampler")

            # Fix mid_block key mappings for attention and residual blocks
            if "mid_block" in k:
                # Map second resnet in mid_block to transformer block's resnet
                if "mid_block.resnets.0" in k:
                    k = k.replace("mid_block.resnets.0", "mid_block.blocks.0.resnet")
                # Map first resnet in mid_block to standalone resnet
                elif "mid_block.resnets.1" in k:
                    k = k.replace("mid_block.resnets.1", "mid_block.resnet")
                # Map attention group norm to block-level attention norm
                elif "mid_block.attentions.0.group_norm" in k:
                    k = k.replace(
                        "mid_block.attentions.0.group_norm",
                        "mid_block.blocks.0.norm_attn",
                    )
                # Map attention layers to transformer block's attention module
                elif "mid_block.attentions.0" in k:
                    k = k.replace("mid_block.attentions.0", "mid_block.blocks.0.attn")

            # Add remapped key-value pair to new state dict
            new_state_dict[k] = v

        # Load the remapped state dict with strict mode to catch missing/unexpected keys
        self.load_state_dict(new_state_dict)


class Decoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.conv_in = nn.Conv2d(config.decoder_in_channels, config.d_model, 3, 1, 1)
        self.mid_block = MidBlock(config)
        up_blocks = []
        out_channels = config.d_model
        for i in reversed(range(config.num_blocks)):
            is_final_block = i == 0
            in_channels = out_channels
            out_channels = config.block_out_channels[i]
            up_blocks.append(
                ResBlock(
                    config,
                    in_channels,
                    out_channels,
                    config.up_layer_resnets,
                    up_sampling=not is_final_block,
                )
            )
        self.up_blocks = nn.Sequential(*up_blocks)
        self.norm = nn.GroupNorm(config.norm_num_groups, out_channels, eps=config.eps)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            config.block_channels, config.decoder_out_channels, 3, 1, 1
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
        state_dict = decoder.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            # Base key replacements for attention layer parameter naming
            k = (
                k.replace("conv_norm_out", "norm")
                .replace("to_k", "Wk")
                .replace("to_q", "Wq")
                .replace("to_v", "Wv")
                .replace("to_out.0", "Wo")
            )

            # Fix upsampler key mapping: upsamplers.0.conv -> upsampler.conv
            if "upsamplers.0.conv" in k:
                k = k.replace("upsamplers.0.conv", "upsampler.conv")
            # Handle other upsamplers keys by removing pluralization
            elif "upsamplers" in k:
                k = k.replace("upsamplers", "upsampler")

            # Fix mid_block key mappings for attention and residual blocks
            if "mid_block" in k:
                # Map second resnet in mid_block to transformer block's resnet
                if "mid_block.resnets.0" in k:
                    k = k.replace("mid_block.resnets.0", "mid_block.blocks.0.resnet")
                # Map first resnet in mid_block to standalone resnet
                elif "mid_block.resnets.1" in k:
                    k = k.replace("mid_block.resnets.1", "mid_block.resnet")
                # Map attention group norm to block-level attention norm
                elif "mid_block.attentions.0.group_norm" in k:
                    k = k.replace(
                        "mid_block.attentions.0.group_norm",
                        "mid_block.blocks.0.norm_attn",
                    )
                # Map attention layers to transformer block's attention module
                elif "mid_block.attentions.0" in k:
                    k = k.replace("mid_block.attentions.0", "mid_block.blocks.0.attn")

            # Add remapped key-value pair to new state dict
            new_state_dict[k] = v
        # Load the remapped state dict
        self.load_state_dict(new_state_dict, strict=True)


class VAE(nn.Module):
    def __init__(self, config: VAEConfig = None):
        super().__init__()
        self.config = config or VAEConfig()
        self.encoder = Encoder(self.config)

        ch = self.config.encoder_out_channels
        self.quant_conv = nn.Conv2d(ch, ch, 1)
        ch = self.config.decoder_in_channels
        self.post_quant_conv = nn.Conv2d(ch, ch, 1)
        self.decoder = Decoder(self.config)

    def encode(self, x):
        x = self.encoder(x)
        x = self.quant_conv(x)
        mean, logvar = torch.chunk(x, 2, dim=1)
        return mean, logvar

    def decode(self, z):
        z = self.post_quant_conv(z)
        x = self.decoder(z)
        return x

    def forward(self, pixel_values, sample=False):
        mean, logvar = self.encode(pixel_values)
        if sample:
            z = mean + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        else:
            z = mean
        pixel_values = self.decode(z)
        return pixel_values

    def load(self, vae: AutoencoderKL):
        self.encoder.load(vae.encoder)
        self.decoder.load(vae.decoder)
        self.quant_conv.load_state_dict(vae.quant_conv.state_dict())
        self.post_quant_conv.load_state_dict(vae.post_quant_conv.state_dict())


# ----- CLIP -----


@dataclass
class CLIPConfig:
    vocab_size: int = 49408
    max_pos_emb: int = 77
    d_model: int = 768
    d_ff: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    eps: float = 1e-5
    dropout: float = 0.0
    is_causal: bool = True

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0
        self.d_k = self.d_model // self.num_heads


class CLIPEmbeddings(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_pos_emb, config.d_model)

    def forward(self, input_ids):
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, device=input_ids.device)
        pos_emb = self.position_embedding(position_ids).unsqueeze(0)
        x = self.token_embedding(input_ids)
        x = x + pos_emb
        return x


class CLIPTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.norm_attn = nn.LayerNorm(d_model, config.eps)
        self.attn = Attention(config)
        self.norm_ff = nn.LayerNorm(d_model, config.eps)
        self.ff = CLIPFF(d_model, config.d_ff)

    def forward(self, x):
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ff(self.norm_ff(x))
        return x


class CLIPFF(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = QuickGELUActivation()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CLIPTextEncoder(nn.Module):
    def __init__(self, config: CLIPConfig = None):
        super().__init__()
        self.config = config or CLIPConfig()
        self.embeddings = CLIPEmbeddings(self.config)
        self.blocks = nn.Sequential(
            *[CLIPTransformerBlock(self.config) for _ in range(self.config.num_layers)]
        )
        self.norm = nn.LayerNorm(self.config.d_model, eps=self.config.eps)

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def load(self, text_encoder: CLIPTextTransformer):
        state_dict = text_encoder.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            k = (
                k.replace("text_model.", "")
                .replace("encoder.layers", "blocks")
                .replace("final_layer_norm", "norm")
                .replace("self_attn", "attn")
                .replace("q_proj", "Wq")
                .replace("k_proj", "Wk")
                .replace("v_proj", "Wv")
                .replace("out_proj", "Wo")
                .replace("mlp", "ff")
                .replace("layer_norm1", "norm_attn")
                .replace("layer_norm2", "norm_ff")
            )

            new_state_dict[k] = v
        self.load_state_dict(new_state_dict)


# ----- UNet -----


@dataclass
class UNetConfig:
    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: Tuple[int] = (320, 640, 1280, 1280)
    max_pos_emb: int = 77
    max_period: int = 10000
    num_layers: int = 2
    num_heads: int = 8
    down_layer_resnets: int = 2
    up_layer_resnets: int = 3
    d_ff_ratio: int = 4
    d_model_cross: int = 768
    padding: int = 1
    norm_num_groups: int = 32
    eps: float = 1e-5
    dropout: float = 0.0

    def __post_init__(self):
        self.num_blocks = len(self.block_out_channels)
        self.block_channels = self.block_out_channels[0]
        self.d_model = self.block_out_channels[-1]
        assert self.block_channels % 2 == 0
        assert self.d_model % self.num_heads == 0
        assert self.d_model_cross % self.num_heads == 0
        self.res_channels = [self.block_channels]
        for i, ch in enumerate(self.block_out_channels):
            self.res_channels.extend([ch] * self.down_layer_resnets)
            if i != self.num_blocks - 1:
                self.res_channels.append(ch)  # for downsampler


class TimeEmbedding(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        self.init_freqs()
        self.fc1 = nn.Linear(config.block_channels, config.d_model)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(config.d_model, config.d_model)

    def init_freqs(self):
        half_dim = self.config.block_channels // 2
        theta = self.config.max_period
        log_freqs = -torch.arange(half_dim, dtype=torch.float32) * math.log(theta)
        log_freqs = log_freqs / half_dim
        freqs = torch.exp(log_freqs)
        self.freqs = freqs

    def forward(self, t):
        device = next(self.parameters()).device
        if not torch.is_tensor(t) or t.ndim == 0:
            if isinstance(t, float):
                dtype = torch.float32
            else:
                dtype = torch.int32
            t = torch.tensor([t], dtype=dtype, device=device)

        freqs = self.freqs.to(device)
        phi = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.cos(phi), torch.sin(phi)], dim=-1)
        emb = emb.to(device, next(self.parameters()).dtype)
        emb = self.fc1(emb)
        emb = self.act(emb)
        emb = self.fc2(emb)
        return emb


class GEGLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels * 2)

    def forward(self, x):
        x = self.fc(x)
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class UNetFF(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.act = GEGLU(d_model, d_ff)
        self.out = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.act(x)
        x = self.out(x)
        return x


class UNetTransformerBlock(nn.Module):
    def __init__(self, config: UNetConfig, d_model, cross_attn):
        super().__init__()
        self.config = config
        self.cross_attn = cross_attn
        d_model_cross = config.d_model_cross if cross_attn else d_model

        self.norm_in = nn.GroupNorm(config.norm_num_groups, d_model, 1e-6)
        self.proj_in = nn.Conv2d(d_model, d_model, 1, 1, 0)
        self.norm_attn1 = nn.LayerNorm(d_model, config.eps)
        self.attn = Attention(config, d_model, bias=False)
        self.norm_attn2 = nn.LayerNorm(d_model, config.eps)
        self.cross_attn = Attention(config, d_model, d_model_cross, bias=False)
        self.norm_ff = nn.LayerNorm(d_model, config.eps)
        self.ff = UNetFF(d_model, config.d_ff_ratio * d_model)
        self.proj_out = nn.Conv2d(d_model, d_model, 1, 1, 0)

    def forward(self, x, x_encoder=None):
        if not self.cross_attn:
            assert x_encoder is None
        B, C, H, W = x.shape
        x0 = x
        x = self.proj_in(self.norm_in(x))
        x = einops.rearrange(x, "B C H W -> B (H W) C")
        x = x + self.attn(self.norm_attn1(x))
        x = x + self.cross_attn(self.norm_attn2(x), x_encoder)
        x = x + self.ff(self.norm_ff(x))
        x = einops.rearrange(x, "B (H W) C -> B C H W", H=H, W=W)
        x = x0 + self.proj_out(x)
        return x


class UnetDownBlock(nn.Module):
    def __init__(
        self,
        config: UNetConfig,
        in_channels,
        out_channels,
        emb_channels,
        res_buf,
        num_layers=2,
        down_sampling=True,
        attn=True,
        cross_attn=True,
    ):
        super().__init__()
        self.res_buf = res_buf

        self.resnets = nn.ModuleList()
        self.transformer_blocks = None
        if attn:
            self.transformer_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(ResNet(config, in_ch, out_channels, emb_channels))
            if self.transformer_blocks is not None:
                self.transformer_blocks.append(
                    UNetTransformerBlock(config, out_channels, cross_attn)
                )

        self.downsampler = None
        if down_sampling:
            self.downsampler = DownSampler(out_channels, out_channels, config.padding)

    def forward(self, x, emb, x_encoder=None):
        for i in range(len(self.resnets)):
            x = self.resnets[i](x, emb)
            if self.transformer_blocks is not None:
                x = self.transformer_blocks[i](x, x_encoder)
            self.res_buf.append(x)

        if self.downsampler is not None:
            x = self.downsampler(x)
            self.res_buf.append(x)
        return x


class UnetMidBlock(nn.Module):
    def __init__(
        self,
        config: UNetConfig,
        in_channels,
        out_channels,
        emb_channels,
        num_layers=1,
        cross_attn=True,
    ):
        super().__init__()

        self.resnets = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(ResNet(config, in_ch, out_channels, emb_channels))
            self.transformer_blocks.append(
                UNetTransformerBlock(config, out_channels, cross_attn)
            )
        self.resnets.append(ResNet(config, out_channels, out_channels, emb_channels))

    def forward(self, x, emb, x_encoder=None):
        for i in range(len(self.transformer_blocks)):
            x = self.resnets[i](x, emb)
            x = self.transformer_blocks[i](x, x_encoder)
        x = self.resnets[-1](x, emb)

        return x


class UnetUpBlock(nn.Module):
    def __init__(
        self,
        config: UNetConfig,
        in_channels,
        out_channels,
        emb_channels,
        res_channels,
        res_buf,
        num_layers=3,
        up_sampling=True,
        attn=True,
        cross_attn=True,
    ):
        super().__init__()
        self.res_buf = res_buf

        self.resnets = nn.ModuleList()
        self.transformer_blocks = None
        if attn:
            self.transformer_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            in_ch += res_channels[i]
            self.resnets.append(ResNet(config, in_ch, out_channels, emb_channels))
            if self.transformer_blocks is not None:
                self.transformer_blocks.append(
                    UNetTransformerBlock(config, out_channels, cross_attn)
                )

        self.upsampler = None
        if up_sampling:
            self.upsampler = UpSampler(out_channels, out_channels)

    def forward(self, x, emb, x_encoder=None):
        for i in range(len(self.resnets)):
            res = self.res_buf.pop(-1)
            x = torch.cat([x, res], dim=1)
            x = self.resnets[i](x, emb)
            if self.transformer_blocks is not None:
                x = self.transformer_blocks[i](x, x_encoder)

        if self.upsampler is not None:
            x = self.upsampler(x)
        return x


class UNet(nn.Module):
    def __init__(self, config: UNetConfig = None):
        super().__init__()
        config = config or UNetConfig()
        self.config = config
        self.res_buf = []
        self.time_embedding = TimeEmbedding(config)
        self.conv_in = nn.Conv2d(config.in_channels, config.block_channels, 3, 1, 1)
        self.init_down_blocks()
        self.init_mid_block()
        self.init_up_blocks()
        self.norm = nn.GroupNorm(
            config.norm_num_groups, config.block_channels, config.eps
        )
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(config.block_channels, config.out_channels, 3, 1, 1)

    def init_down_blocks(self):
        config = self.config
        self.down_blocks = nn.ModuleList()

        out_channels = config.block_channels
        for i in range(config.num_blocks):
            in_channels = out_channels
            is_final_block = i == config.num_blocks - 1
            out_channels = config.block_out_channels[i]
            block = UnetDownBlock(
                config,
                in_channels,
                out_channels,
                config.d_model,
                res_buf=self.res_buf,
                down_sampling=not is_final_block,
                attn=not is_final_block,
            )
            self.down_blocks.append(block)

    def init_mid_block(self):
        d_model = self.config.d_model
        self.mid_block = UnetMidBlock(self.config, d_model, d_model, d_model)

    def init_up_blocks(self):
        config = self.config
        self.up_blocks = nn.ModuleList()

        out_channels = config.d_model
        res_channels = config.res_channels
        for i in reversed(range(config.num_blocks)):
            in_channels = out_channels
            is_first_block = i == config.num_blocks - 1
            is_final_block = i == 0
            out_channels = config.block_out_channels[i]
            block = UnetUpBlock(
                config,
                in_channels,
                out_channels,
                config.d_model,
                res_channels=res_channels[-config.up_layer_resnets :][::-1],
                res_buf=self.res_buf,
                num_layers=config.up_layer_resnets,
                up_sampling=not is_final_block,
                attn=not is_first_block,
            )
            self.up_blocks.append(block)
            res_channels = res_channels[: -config.up_layer_resnets]

    def forward(self, x, t, x_encoder):
        self.clean_res_buf()
        emb = self.time_embedding(t)
        x = self.conv_in(x)
        self.res_buf.append(x)
        for down_block in self.down_blocks:
            x = down_block(x, emb, x_encoder)
        x = self.mid_block(x, emb, x_encoder)
        for up_block in self.up_blocks:
            x = up_block(x, emb, x_encoder)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv_out(x)
        return x

    def clean_res_buf(self):
        while len(self.res_buf) > 0:
            res = self.res_buf.pop()
            del res

    def load(self, unet: UNet2DConditionModel):
        state_dict = unet.state_dict()
        new_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # 1. Remove the extra "transformer_blocks.0" level that appears after attentions
            # SD: down_blocks.0.attentions.0.transformer_blocks.0.norm1.weight
            # Target: down_blocks.0.transformer_blocks.0.norm_attn1.weight
            pattern = r"\.attentions\.(\d+)\.transformer_blocks\.0\."
            new_key = re.sub(pattern, r".transformer_blocks.\1.", new_key)

            # 2. For mid_block, do similar transformation
            if "mid_block.attentions.0.transformer_blocks.0." in new_key:
                new_key = new_key.replace(
                    "mid_block.attentions.0.transformer_blocks.0.",
                    "mid_block.transformer_blocks.0.",
                )

            # 3. Basic module name replacements (preserve resnet norm layers)
            new_key = new_key.replace("time_embedding.linear_", "time_embedding.fc")
            new_key = new_key.replace("conv_norm_out", "norm")
            new_key = new_key.replace("time_emb_proj", "emb_proj")

            # 4. Attention module transformations
            new_key = new_key.replace("to_q", "Wq")
            new_key = new_key.replace("to_k", "Wk")
            new_key = new_key.replace("to_v", "Wv")

            if "to_out.0" in new_key:
                new_key = new_key.replace("to_out.0", "Wo")

            # 5. Handle norm layer naming differences
            # Only apply to transformer block norms, NOT resnet norms
            if ".norm1." in new_key and "transformer_blocks" in new_key:
                new_key = new_key.replace(".norm1.", ".norm_attn1.")
            elif ".norm2." in new_key and "transformer_blocks" in new_key:
                new_key = new_key.replace(".norm2.", ".norm_attn2.")
            elif ".norm3." in new_key and "transformer_blocks" in new_key:
                new_key = new_key.replace(".norm3.", ".norm_ff.")

            # 6. Transform attention type naming
            if ".attn1." in new_key:
                new_key = new_key.replace(".attn1.", ".attn.")
            elif ".attn2." in new_key:
                new_key = new_key.replace(".attn2.", ".cross_attn.")

            # 7. Transform FFN layers
            if "ff.net.0" in new_key:
                new_key = new_key.replace("ff.net.0", "ff.act.fc")
                # Remove .proj suffix
                if ".proj." in new_key:
                    new_key = new_key.replace(".proj.", ".")
            elif "ff.net.2" in new_key:
                new_key = new_key.replace("ff.net.2", "ff.out")

            # 8. Handle downsamplers/upsamplers
            if "downsamplers.0" in new_key:
                new_key = new_key.replace("downsamplers.0", "downsampler")
            elif "upsamplers.0" in new_key:
                new_key = new_key.replace("upsamplers.0", "upsampler")

            # 9. Fix norm -> norm_in for transformer block input norm
            # Only apply to transformer block norm layers
            if "transformer_blocks" in new_key and ".norm.weight" in new_key:
                # Check if this is the input norm (not attn1, attn2, or ff norm)
                if "norm_attn" not in new_key and "norm_ff" not in new_key:
                    new_key = new_key.replace(".norm.", ".norm_in.")

            # 10. Special handling for attention block projections
            # Map attentions.X.proj_in -> transformer_blocks.X.proj_in
            proj_in_pattern = r"\.attentions\.(\d+)\.proj_in\."
            new_key = re.sub(
                proj_in_pattern, r".transformer_blocks.\1.proj_in.", new_key
            )

            # Map attentions.X.proj_out -> transformer_blocks.X.proj_out
            proj_out_pattern = r"\.attentions\.(\d+)\.proj_out\."
            new_key = re.sub(
                proj_out_pattern, r".transformer_blocks.\1.proj_out.", new_key
            )

            # Map mid_block attentions projections
            if "mid_block.attentions.0.proj_in." in new_key:
                new_key = new_key.replace(
                    "mid_block.attentions.0.proj_in.",
                    "mid_block.transformer_blocks.0.proj_in.",
                )
            if "mid_block.attentions.0.proj_out." in new_key:
                new_key = new_key.replace(
                    "mid_block.attentions.0.proj_out.",
                    "mid_block.transformer_blocks.0.proj_out.",
                )

            # 11. Fix attention norm mappings
            # Map attentions.X.norm -> transformer_blocks.X.norm_in
            norm_pattern = r"\.attentions\.(\d+)\.norm\."
            new_key = re.sub(norm_pattern, r".transformer_blocks.\1.norm_in.", new_key)

            # Fix mid_block attention norm
            if "mid_block.attentions.0.norm." in new_key:
                new_key = new_key.replace(
                    "mid_block.attentions.0.norm.",
                    "mid_block.transformer_blocks.0.norm_in.",
                )

            # Store the converted key
            new_state_dict[new_key] = value

        self.load_state_dict(new_state_dict)


# ----- diffusion -----


class StableDiffusion(nn.Module):
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        scheduler: KarrasDiffusionSchedulers,
        vae_config: VAEConfig = None,
        clip_config: CLIPConfig = None,
        unet_config: UNetConfig = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.vae_config = vae_config or VAEConfig()
        self.clip_config = clip_config or CLIPConfig()
        self.unet_config = unet_config or UNetConfig()
        self.vae = VAE(self.vae_config)
        self.text_encoder = CLIPTextEncoder(self.clip_config)
        self.unet = UNet(self.unet_config)
        self.vae_scale_factor = 1
        for upblock in self.unet.up_blocks:
            if upblock.upsampler is not None:
                self.vae_scale_factor *= 2
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def encode_text(self, text):
        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        embs = self.text_encoder(input_ids.to(self.device))
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

        prompt_embeds = self.encode_text(prompts)
        negative_prompt_embeds = self.encode_text(negative_prompts)
        return prompt_embeds, negative_prompt_embeds

    @torch.inference_mode()
    def forward(
        self,
        prompts,
        height=512,
        width=512,
        num_steps=50,
        negative_prompts=None,
        guidance_scale=7.5,
    ):
        device = self.device
        dtype = self.dtype

        # ----- embed prompts -----
        prompt_embeds, neg_prompt_embeds = self.encode_prompt(prompts, negative_prompts)
        batch_size = prompt_embeds.shape[0]
        prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds])

        # ----- sample latents -----
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        latent_channels = self.vae_config.latent_channels
        latent_shape = (batch_size, latent_channels, latent_height, latent_width)
        latents = torch.randn(latent_shape, device=device, dtype=dtype)

        # ----- prepare time steps -----
        self.scheduler.set_timesteps(num_steps, device=device)
        ts = self.scheduler.timesteps

        # ----- denoise -----
        for i, t in enumerate(tqdm(ts)):
            eps_pred = self.unet(latents.repeat(2, 1, 1, 1), t, prompt_embeds)
            eps_uncond, eps_cond = eps_pred.chunk(2)
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            latents = self.scheduler.step(eps_pred, t, latents, return_dict=False)[0]

        images = self.vae.decode(latents / self.vae.config.scaling_factor)
        images = self.image_processor.postprocess(
            images, output_type="pil", do_denormalize=[True] * batch_size
        )
        return images

    def load(self, sd_pipeline: StableDiffusionPipeline):
        self.vae.load(sd_pipeline.vae)
        self.text_encoder.load(sd_pipeline.text_encoder)
        self.unet.load(sd_pipeline.unet)
