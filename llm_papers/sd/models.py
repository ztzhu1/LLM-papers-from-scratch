from dataclasses import dataclass

from diffusers import AutoencoderKL, StableDiffusionPipeline
import einops
import torch
from torch import nn
import torch.nn.functional as F

# ----- VAE -----


@dataclass
class Config:
    in_channels: int = 3
    out_channels: int = 3
    block_channels: int = 128
    num_down_blocks: int = 4
    layer_resnets: int = 2
    layer_attns: int = 1
    num_heads: int = 1
    norm_num_groups: int = 32
    eps: float = 1e-6
    dropout: float = 0.0

    def __post_init__(self):
        assert self.block_channels % self.norm_num_groups == 0
        self.block_out_channels = []
        for i in range(self.num_down_blocks - 1):
            out_ch = self.block_channels * (2**i)
            self.block_out_channels.append(out_ch)
        self.d_model = self.block_out_channels[-1]
        assert self.d_model % self.num_heads == 0
        self.d_k = self.d_model // self.num_heads


class ResNet(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        self.config = config
        self.norm1 = nn.GroupNorm(config.norm_num_groups, in_channels, eps=config.eps)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(config.norm_num_groups, in_channels, eps=config.eps)
        self.dropout = nn.Dropout(config.dropout)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        inputs = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return inputs + x


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.norm = nn.GroupNorm(config.norm_num_groups, d_model, eps=config.eps)
        self.Wqkv = nn.Linear(d_model, 3 * config.num_heads * config.d_k)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        inputs = x
        height, width = x.shape[2], x.shape[3]
        x = self.norm(x)
        x = einops.rearrange(x, "B C H W -> B (H W) C")
        QKV = self.Wqkv(x)
        QKV = einops.rearrange(QKV, "B L (qkv head d_k) -> qkv B head L d_k", qkv=3)
        Q, K, V = QKV.contiguous().unbind(0)
        x = F.scaled_dot_product_attention(Q, K, V)
        x = self.Wo(x)
        x = self.dropout(x)
        x = einops.rearrange(x, "B (H W) C -> B C H W", H=height, W=width)
        return inputs + x


class DownBlock(nn.Module):
    def __init__(self, config, in_channels, out_channels, down_sampling):
        super().__init__()
        self.config = config
        self.resnets = []
        for i in range(config.layer_resnets):
            if i == 0:
                self.resnets.append(ResNet(config, in_channels, out_channels))
            else:
                self.resnets.append(ResNet(config, out_channels, out_channels))
        self.resnets = nn.Sequential(*self.resnets)
        if down_sampling:
            self.down_sampler = nn.Conv2d(out_channels, out_channels, 3, 2, 0)
        else:
            self.down_sampler = nn.Identity()

    def forward(self, x):
        x = self.resnets(x)
        x = self.down_sampler(F.pad(x, (0, 1, 0, 1)))
        return x


class MidBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.attns = nn.Sequential(
            *[Attention(config) for _ in range(config.layer_attns)]
        )
        ch = config.block_out_channels[-1]
        self.resnets = nn.Sequential(
            *[ResNet(config, ch, ch) for _ in range(config.layer_resnets)]
        )

    def forward(self, x):
        x = self.attns(x)
        x = self.resnets(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        if config is None:
            config = Config()
        self.config = config

        self.conv_in = nn.Conv2d(config.in_channels, config.block_channels, 3, 1, 1)
        down_blocks = []
        out_channels = config.block_channels
        for i in range(config.num_down_blocks):
            in_channels = out_channels
            is_final_block = i == config.num_down_blocks - 1
            if not is_final_block:
                out_channels = config.block_out_channels[i]
            block = DownBlock(config, in_channels, out_channels, not is_final_block)
            down_blocks.append(block)
        self.down_blocks = nn.Sequential(*down_blocks)
        self.mid_block = MidBlock(config)
        self.norm = nn.GroupNorm(config.norm_num_groups, out_channels, eps=config.eps)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down_blocks(x)
        x = self.mid_block(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        return x


class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        return x
