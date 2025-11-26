import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        drop_rate=0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.Wqkv = nn.Linear(d_model, d_model * 3)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            attention: (batch_size, seq_len, d_model)
        """
        B, L, D = x.shape
        H = self.num_heads
        d_k = self.d_k
        QKV = self.Wqkv(x).view(B, L, 3, H, d_k).swapaxes(1, 3)  # (B, H, 3, L, d_k)
        Q, K, V = QKV.unbind(2)  # (B, H, L, d_k)
        multi_head = F.scaled_dot_product_attention(Q, K, V)  # (B, H, L, d_k)
        multi_head = multi_head.swapaxes(1, 2).flatten(2)  # (B, L, D)
        attention = self.Wo(multi_head)
        return attention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_mlp):
        super().__init__()

        self.attn = Attention(d_model, num_heads)
        self.norm_attn = nn.LayerNorm(d_model, 1e-6)
        self.mlp = MLP(
            in_features=d_model,
            hidden_features=d_mlp,
            out_features=d_model,
        )
        self.norm_mlp = nn.LayerNorm(d_model, 1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 32,
        in_chans: int = 3,
        num_classes: int = 100,
        d_model: int = 768,
        d_mlp: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        num_patches = (img_size // patch_size) ** 2
        num_tokens = 1 + num_patches
        self.patch_embed = nn.Conv2d(in_chans, d_model, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.empty(1, num_tokens, d_model))
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, num_heads, d_mlp) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model, 1e-6)
        self.head = nn.Linear(d_model, num_classes)
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)

    def cat_cls(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_patches, d_model)

        Returns:
            x: (batch_size, num_tokens, d_model)
        """
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], 1)
        return x

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_tokens, d_model)

        Returns:
            x: (batch_size, d_model)
        """
        return x[:, 0]  # cls_token

    def forward_features(self, x):
        x = self.patch_embed(x)  # (B, d_model, H, W)
        x = x.flatten(2).swapaxes(1, 2)  # (B, H*W, d_model)
        x = self.cat_cls(x)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x):
        x = self.pool(x)
        logits = self.head(x)
        return logits

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def load_timm(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            k = (
                k.replace("patch_embed.proj", "patch_embed")
                .replace("qkv", "Wqkv")
                .replace("attn.proj", "attn.Wo")
                .replace("norm1", "norm_attn")
                .replace("norm2", "norm_mlp")
            )
            new_state_dict[k] = v
        self.load_state_dict(new_state_dict)
