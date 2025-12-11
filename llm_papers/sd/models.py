from dataclasses import dataclass
import re

from diffusers import AutoencoderKL, StableDiffusionPipeline
import einops
import torch
from torch import nn
import torch.nn.functional as F

# ----- VAE -----


@dataclass
class Config:
    in_channels: int = 3
    encoder_out_channels: int = 8
    block_channels: int = 128
    num_down_blocks: int = 4
    num_trans_blocks: int = 1
    layer_resnets: int = 2
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
        self.norm2 = nn.GroupNorm(config.norm_num_groups, out_channels, eps=config.eps)
        self.dropout = nn.Dropout(config.dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward_proj(self, x):
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        if self.conv_shortcut is not None:
            x0 = self.conv_shortcut(x)
        else:
            x0 = x
        return x0 + self.forward_proj(self.norm1(x))


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.norm = nn.GroupNorm(config.norm_num_groups, d_model, eps=config.eps)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward_attn(self, x):
        height, width = x.shape[2], x.shape[3]
        h = self.config.num_heads
        d_k = self.config.d_k

        x = einops.rearrange(x, "B C H W -> B (H W) C")

        Q = einops.rearrange(
            self.Wq(x), "B L (head d_k) -> B head L d_k", head=h, d_k=d_k
        )
        K = einops.rearrange(
            self.Wk(x), "B L (head d_k) -> B head L d_k", head=h, d_k=d_k
        )
        V = einops.rearrange(
            self.Wv(x), "B L (head d_k) -> B head L d_k", head=h, d_k=d_k
        )
        x = F.scaled_dot_product_attention(Q, K, V)
        x = einops.rearrange(x, "B head L d_k -> B L (head d_k)")
        x = self.Wo(x)

        x = self.dropout(x)
        x = einops.rearrange(x, "B (H W) C -> B C H W", H=height, W=width)

        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_attn(self.norm(x)) + x
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.attn = Attention(config)
        self.resnet = ResNet(config, d_model, d_model)

    def forward(self, x):
        x = self.attn(x)
        x = self.resnet(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, config: Config, in_channels, out_channels, down_sampling):
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
            self.downsamplers = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 2, 0)
            )
        else:
            self.downsamplers = None

    def forward(self, x):
        x = self.resnets(x)
        if self.downsamplers is not None:
            x = self.downsamplers(F.pad(x, (0, 1, 0, 1)))
        return x


class MidBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.resnet = ResNet(config, config.d_model, config.d_model)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_trans_blocks)]
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.blocks(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
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
        self.conv_out = nn.Conv2d(out_channels, config.encoder_out_channels, 3, 1, 1)

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
            k = (
                k.replace("conv_norm_out", "norm")
                .replace("to_k", "Wk")
                .replace("to_q", "Wq")
                .replace("to_v", "Wv")
                .replace("to_out.0", "Wo")
            )
            if "downsamplers" in k:
                k = k.replace("conv.", "")
            if "attentions" in k:
                k = k.replace("group_norm", "norm")
            if "mid_block" in k:
                if "resnets.0" in k:
                    k = k.replace("resnets.0", "resnet")
                else:
                    name, layer, others = re.fullmatch(
                        r"mid_block\.(.*?)\.(\d+)(.*?)", k
                    ).groups()
                    layer = int(layer)
                    if name == "attentions":
                        k = f"mid_block.blocks.{layer}.attn{others}"
                    elif name == "resnets":
                        k = f"mid_block.blocks.{layer-1}.resnet{others}"
                    else:
                        raise ValueError(f"Unknown layer name: {name}")

            new_state_dict[k] = v
        self.load_state_dict(new_state_dict)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        return x


class VAE(nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        if config is None:
            config = Config()
        self.config = config
        self.encoder = Encoder(config)

    def forward(self, x):
        x = self.encoder(x)
        return x

    def load(self, vae: AutoencoderKL):
        self.encoder.load(vae.encoder)
