"""Vision Transformer (ViT), torchvision-style, for the modernized jittor model
zoo. Pure jittor.nn so it runs on both NVIDIA (CUDA) and Ascend (NPU) and under
`import jittor as torch`. Patch-embed (conv) + class token + learned pos-embed +
pre-norm transformer encoder.

    from jittor.models import vit_b_16
    m = vit_b_16(num_classes=1000); y = m(jt.randn(2,3,224,224))
"""
import math
import jittor as jt
from jittor import nn

__all__ = ["VisionTransformer", "vit_b_16", "vit_b_32", "vit_l_16"]


class _MLP(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        return self.drop(self.fc2(self.drop(nn.gelu(self.fc1(x)))))


class _Attention(nn.Module):
    def __init__(self, dim, num_heads, drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = nn.softmax(jt.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale, dim=-1)
        y = jt.matmul(att, v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.drop(self.proj(y))


class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads, drop)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), drop)

    def execute(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000,
                 dim=768, depth=12, num_heads=12, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        assert image_size % patch_size == 0
        n_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = jt.zeros((1, 1, dim))
        self.pos_embed = jt.zeros((1, n_patches + 1, dim))
        nn.init.gauss_(self.pos_embed, 0.0, 0.02)
        nn.init.gauss_(self.cls_token, 0.0, 0.02)
        self.blocks = nn.ModuleList([_Block(dim, num_heads, mlp_ratio, drop) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def execute(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                       # B, dim, H', W'
        x = x.reshape(B, x.shape[1], -1).transpose(0, 2, 1)   # B, N, dim
        cls = self.cls_token.broadcast([B, 1, x.shape[2]])
        x = jt.concat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        return self.head(x[:, 0])                      # class token


def vit_b_16(num_classes=1000, **kw):
    return VisionTransformer(patch_size=16, dim=768, depth=12, num_heads=12,
                             num_classes=num_classes, **kw)


def vit_b_32(num_classes=1000, **kw):
    return VisionTransformer(patch_size=32, dim=768, depth=12, num_heads=12,
                             num_classes=num_classes, **kw)


def vit_l_16(num_classes=1000, **kw):
    return VisionTransformer(patch_size=16, dim=1024, depth=24, num_heads=16,
                             num_classes=num_classes, **kw)
