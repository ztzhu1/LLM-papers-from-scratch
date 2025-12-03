from dataclasses import dataclass
import math
from pathlib import Path

from PIL import Image
import datasets
import einops
import timm
from timm.models import VisionTransformer
import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    RobertaModel,
    RobertaTokenizerFast,
    TimmWrapperImageProcessor,
)

from llm_papers.utils import device

# ----- utils -----


def load_tokenizer(model_name="distilroberta-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def load_image_processor():
    image_processor = AutoImageProcessor.from_pretrained(
        "timm/vit_base_patch32_224.augreg_in21k_ft_in1k"
    )
    return image_processor


def load_pretrained_roberta() -> tuple[RobertaTokenizerFast, RobertaModel]:
    model_name = "distilroberta-base"
    tokenizer = load_tokenizer(model_name)
    text_model = AutoModel.from_pretrained(model_name).eval()
    return tokenizer, text_model


def load_pretrained_vit() -> tuple[TimmWrapperImageProcessor, VisionTransformer]:
    image_processor = load_image_processor()
    vision_model = timm.create_model("vit_base_patch32_224", pretrained=True).eval()
    return image_processor, vision_model


def long_arange(size):
    return torch.arange(size, device=device, dtype=torch.long)


# ----- for both models -----


class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.Wqkv = nn.Linear(d_model, 3 * d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_tokens, d_model)
            attention_mask: (batch_size, num_tokens)
        """
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = attention_mask[:, None, :, None]
            attention_mask = None
        QKV = self.Wqkv(x)  # (batch_size, num_tokens, 3*d_model)
        QKV = einops.rearrange(
            QKV,
            "B L (qkv num_heads d_k) -> qkv B num_heads L d_k",
            qkv=3,
            num_heads=self.num_heads,
            d_k=self.d_k,
        ).contiguous()
        Q, K, V = QKV.unbind(0)  # (batch_size, num_heads, num_tokens, d_k)
        attentions = F.scaled_dot_product_attention(Q, K, V, attention_mask)
        attentions = einops.rearrange(
            attentions, "B num_heads L d_k -> B L (num_heads d_k)"
        )
        attentions = self.Wo(attentions)
        return attentions


class MLP(nn.Module):
    def __init__(self, in_channels, d_mlp):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, d_mlp)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_mlp, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.d_model
        d_mlp = config.d_mlp
        num_heads = config.num_heads
        eps = config.eps
        self.norm_attn = nn.LayerNorm(d_model, eps)
        self.attn = Attention(d_model, num_heads)
        self.norm_mlp = nn.LayerNorm(d_model, eps)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        if self.config.post_norm:
            x = self.norm_attn(x + self.attn(x, attention_mask))
            x = self.norm_mlp(x + self.mlp(x))
        else:
            x = x + self.attn(self.norm_attn(x), attention_mask)
            x = x + self.mlp(self.norm_mlp(x))
        return x


# ----- text model -----


@dataclass
class RobertaConfig:
    vocab_size = 50265
    type_vocab_size = 1
    max_seq_len = 514
    pad_token_id = 1
    d_model = 768
    d_mlp = 3072
    num_heads = 12
    num_layers = 6
    eps = 1e-5
    post_norm = True

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0


class RobertaEmbeddings(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        vocab_size = config.vocab_size
        max_seq_len = config.max_seq_len
        pad_token_id = config.pad_token_id
        self.word_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embed = nn.Embedding(max_seq_len, d_model, padding_idx=pad_token_id)
        if config.type_vocab_size > 1:
            padding_idx = None
        else:
            padding_idx = 0
        self.type_embed = nn.Embedding(
            config.type_vocab_size, d_model, padding_idx=padding_idx
        )
        self.register_buffer(
            "type_ids", torch.zeros(max_seq_len, dtype=torch.long), persistent=False
        )
        self.norm = nn.LayerNorm(d_model, config.eps)

    def create_pos_ids(self, x):
        mask = (x != self.config.pad_token_id).int()
        pos_ids = torch.cumsum(mask, dim=1) * mask + self.config.pad_token_id
        return pos_ids.long()

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len)
        """
        seq_len = x.shape[1]
        pos_ids = self.create_pos_ids(x)
        type_ids = self.type_ids[:seq_len]
        x = self.word_embed(x) + self.pos_embed(pos_ids) + self.type_embed(type_ids)
        x = self.norm(x)
        return x


class Roberta(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_layers)])

    def pool(self, x):
        return x[:, 0]

    def forward_features(self, x, attention_mask=None):
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x, attention_mask)
        return x

    def forward(self, input_ids, attention_mask=None):
        x = self.forward_features(input_ids, attention_mask)
        x = self.pool(x)
        return x

    def load(self, pretrained_model):
        state_dict = pretrained_model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            if "pooler" in k or "encoder" in k:
                continue
            k = (
                k.replace("word_embeddings", "word_embed")
                .replace("position_embeddings", "pos_embed")
                .replace("token_type_embeddings", "type_embed")
                .replace("embeddings.LayerNorm", "embeddings.norm")
            )

            new_state_dict[k] = v
        for layer in range(self.config.num_layers):
            old_prefix = f"encoder.layer.{layer}"
            new_prefix = f"blocks.{layer}"
            for suffix in ["weight", "bias"]:
                new_state_dict[f"{new_prefix}.attn.Wqkv.{suffix}"] = torch.cat(
                    [
                        state_dict[f"{old_prefix}.attention.self.{type}.{suffix}"]
                        for type in ["query", "key", "value"]
                    ],
                    0,
                )
                new_state_dict[f"{new_prefix}.attn.Wo.{suffix}"] = state_dict[
                    f"{old_prefix}.attention.output.dense.{suffix}"
                ]
                new_state_dict[f"{new_prefix}.norm_attn.{suffix}"] = state_dict[
                    f"{old_prefix}.attention.output.LayerNorm.{suffix}"
                ]
                new_state_dict[f"{new_prefix}.mlp.fc1.{suffix}"] = state_dict[
                    f"{old_prefix}.intermediate.dense.{suffix}"
                ]
                new_state_dict[f"{new_prefix}.mlp.fc2.{suffix}"] = state_dict[
                    f"{old_prefix}.output.dense.{suffix}"
                ]
                new_state_dict[f"{new_prefix}.norm_mlp.{suffix}"] = state_dict[
                    f"{old_prefix}.output.LayerNorm.{suffix}"
                ]
        self.load_state_dict(new_state_dict)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


# ----- vision model -----


@dataclass
class ViTConfig:
    img_size = 224
    channels = 3
    patch_size = 32
    d_model = 768
    d_mlp = 3072
    num_heads = 12
    num_layers = 12
    eps = 1e-6
    post_norm = False

    def __post_init__(self):
        assert self.img_size % self.patch_size == 0
        assert self.d_model % self.num_heads == 0
        self.num_patchs = (self.img_size // self.patch_size) ** 2
        self.num_tokens = self.num_patchs + 1


class ViTPatchEmbed(nn.Module):
    def __init__(self, patch_size, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, patch_size, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, img_size, img_size)
        """
        x = self.proj(x)  # (batch_size, out_channels, num_patches, num_patches)
        x = einops.rearrange(x, "b c ph pw -> b (ph pw) c")
        return x


class ViTPosEmbed(nn.Module):
    def __init__(self, d_pos_embed, d_model):
        super().__init__()
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.empty(1, d_pos_embed, d_model))
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_patches, d_model)

        Returns:
            x: (batch_size, num_tokens, d_model)
        """
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        return x


class ViT(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.patch_embed = ViTPatchEmbed(config.patch_size, config.channels, d_model)
        self.pos_embed = ViTPosEmbed(self.config.num_tokens, d_model)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(d_model, config.eps)

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_tokens, d_model)
        """
        return x[:, 0]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        """
        x: (batch_size, in_channels, img_size, img_size)
        """
        x = self.forward_features(x)
        x = self.pool(x)
        return x

    def load(self, pretrained_model):
        state_dict = pretrained_model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            if "head" in k:
                continue
            k = (
                k.replace("timm_model.", "")
                .replace("pos_embed", "pos_embed.pos_embed")
                .replace("cls_token", "pos_embed.cls_token")
                .replace("qkv", "Wqkv")
                .replace("attn.proj", "attn.Wo")
                .replace("norm1", "norm_attn")
                .replace("norm2", "norm_mlp")
            )
            new_state_dict[k] = v
        self.load_state_dict(new_state_dict)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


# ----- CLIP model -----


class CLIP(nn.Module):
    def __init__(
        self,
        d_proj=512,
        init_logit_scale=math.log(1 / 0.07),
        text_config: RobertaConfig = None,
        vision_config: ViTConfig = None,
    ):
        super().__init__()
        self.d_proj = d_proj
        text_config = text_config or RobertaConfig()
        vision_config = vision_config or ViTConfig()
        self.text_config = text_config
        self.vision_config = vision_config

        self.text_model = Roberta(text_config)
        self.vision_model = ViT(vision_config)

        self.text_proj = nn.Linear(text_config.d_model, d_proj)
        self.vision_proj = nn.Linear(vision_config.d_model, d_proj)
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
        self.init_weights()

        self.eval_text_embed_cache = None

    def init_weights(self):
        nn.init.xavier_normal_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)
        nn.init.xavier_normal_(self.vision_proj.weight)
        nn.init.zeros_(self.vision_proj.bias)

    def encode_text(self, input_ids, attention_mask=None):
        text_embed = self.text_model(input_ids, attention_mask)
        text_embed = self.text_proj(text_embed)
        text_embed = F.normalize(text_embed, dim=-1)
        return text_embed

    def encode_image(self, pixel_values):
        image_embed = self.vision_model(pixel_values)
        image_embed = self.vision_proj(image_embed)
        image_embed = F.normalize(image_embed, dim=-1)
        return image_embed

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        if self.training:
            text_embed = self.encode_text(input_ids, attention_mask)
            text_targets = long_arange(input_ids.size(0))
            image_targets = long_arange(pixel_values.size(0))
        else:
            if self.eval_text_embed_cache is None:
                self.eval_text_embed_cache = self.encode_text(input_ids, attention_mask)
            text_embed = self.eval_text_embed_cache
            text_targets = None
            image_targets = labels
        image_embed = self.encode_image(pixel_values)

        similarity = einops.einsum(
            text_embed, image_embed, "bt d_proj, bi d_proj -> bt bi"
        )
        text_logits = similarity * self.logit_scale.exp()
        image_logits = text_logits.T

        image_loss = F.cross_entropy(image_logits, image_targets)
        if self.training:
            text_loss = F.cross_entropy(text_logits, text_targets)
            loss = (text_loss + image_loss) / 2
        else:
            loss = image_loss
        return loss, text_logits, image_logits

    def load(self, pretrained_text_model, pretrained_vision_model):
        self.text_model.load(pretrained_text_model)
        self.vision_model.load(pretrained_vision_model)

    def freeze(self):
        self.text_model.freeze()
        self.vision_model.freeze()

    def train(self, mode: bool = True):
        super().train(mode)
        self.eval_text_embed_cache = None
