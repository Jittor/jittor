"""Matched Jittor and PyTorch reference networks for end-to-end parity.

The repository already compares the torchvision-style model zoo forward pass
against PyTorch.  These builders extend that to the architectures named in the
2.0 goals -- ResNet, ViT, GPT-2 and a diffusion UNet -- and to the *backward*
pass, which the zoo comparison never exercised.

Each entry returns two structurally identical modules, one written against
``jittor.nn`` and one against ``torch.nn``.  Weights are transferred from the
PyTorch module so any observed difference comes from operator semantics rather
than initialization.  The definitions are deliberately small: parity is an
operator-coverage question, and a 12-layer ViT only makes the CPU gate slow
without covering a single additional operator.
"""

import math

import jittor as jt
from jittor import nn as jnn


# --------------------------------------------------------------------- helpers

def _torch_modules():
    """Import the independent PyTorch oracle lazily, as the suite requires."""
    from _helpers.torch_runtime import import_torch_modules

    return import_torch_modules("torch", "torch.nn", "torch.nn.functional")


# ------------------------------------------------------------------------ ViT

def _jittor_vit(dim, depth, heads, patch, image, classes):
    from jittor.models.vision_transformer import VisionTransformer

    return VisionTransformer(
        image_size=image,
        patch_size=patch,
        num_classes=classes,
        dim=dim,
        depth=depth,
        num_heads=heads,
        drop=0.0,
    )


def _torch_vit(torch, nn, dim, depth, heads, patch, image, classes):
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * 4)
            self.fc2 = nn.Linear(dim * 4, dim)

        def forward(self, x):
            return self.fc2(nn.functional.gelu(self.fc1(x)))

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = heads
            self.head_dim = dim // heads
            self.scale = self.head_dim ** -0.5
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)

        def forward(self, x):
            b, n, c = x.shape
            qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            att = (q @ k.transpose(-2, -1)) * self.scale
            att = att.softmax(dim=-1)
            y = (att @ v).transpose(1, 2).reshape(b, n, c)
            return self.proj(y)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(dim)
            self.attn = Attention()
            self.ln2 = nn.LayerNorm(dim)
            self.mlp = MLP()

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x))

    class ViT(nn.Module):
        def __init__(self):
            super().__init__()
            n_patches = (image // patch) ** 2
            self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
            nn.init.normal_(self.pos_embed, 0.0, 0.02)
            nn.init.normal_(self.cls_token, 0.0, 0.02)
            self.blocks = nn.ModuleList([Block() for _ in range(depth)])
            self.ln = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, classes)

        def forward(self, x):
            b = x.shape[0]
            x = self.patch_embed(x)
            x = x.reshape(b, x.shape[1], -1).transpose(1, 2)
            cls = self.cls_token.expand(b, -1, -1)
            x = torch.cat([cls, x], dim=1) + self.pos_embed
            for block in self.blocks:
                x = block(x)
            return self.head(self.ln(x)[:, 0])

    return ViT()


# ---------------------------------------------------------------------- GPT-2

def _jittor_gpt2(vocab, block, dim, depth, heads):
    class CausalSelfAttention(jnn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = heads
            self.head_dim = dim // heads
            self.scale = self.head_dim ** -0.5
            self.c_attn = jnn.Linear(dim, dim * 3)
            self.c_proj = jnn.Linear(dim, dim)
            mask = jt.tril(jt.ones((block, block)))
            self.mask = mask.reshape(1, 1, block, block)
            self.mask.requires_grad = False

        def execute(self, x):
            b, t, c = x.shape
            qkv = self.c_attn(x).reshape(b, t, 3, self.num_heads, self.head_dim)
            qkv = qkv.transpose(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            att = jt.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
            keep = self.mask[:, :, :t, :t]
            att = att * keep + (keep - 1.0) * 1e9
            att = jnn.softmax(att, dim=-1)
            y = jt.matmul(att, v).transpose(0, 2, 1, 3).reshape(b, t, c)
            return self.c_proj(y)

    class MLP(jnn.Module):
        def __init__(self):
            super().__init__()
            self.c_fc = jnn.Linear(dim, dim * 4)
            self.c_proj = jnn.Linear(dim * 4, dim)

        def execute(self, x):
            return self.c_proj(jnn.gelu(self.c_fc(x)))

    class Block(jnn.Module):
        def __init__(self):
            super().__init__()
            self.ln_1 = jnn.LayerNorm(dim)
            self.attn = CausalSelfAttention()
            self.ln_2 = jnn.LayerNorm(dim)
            self.mlp = MLP()

        def execute(self, x):
            x = x + self.attn(self.ln_1(x))
            return x + self.mlp(self.ln_2(x))

    class GPT2(jnn.Module):
        def __init__(self):
            super().__init__()
            self.wte = jnn.Embedding(vocab, dim)
            self.wpe = jnn.Embedding(block, dim)
            self.h = jnn.ModuleList([Block() for _ in range(depth)])
            self.ln_f = jnn.LayerNorm(dim)
            self.lm_head = jnn.Linear(dim, vocab, bias=False)

        def execute(self, idx):
            b, t = idx.shape
            pos = jt.arange(t).broadcast([b, t])
            x = self.wte(idx) + self.wpe(pos)
            for layer in self.h:
                x = layer(x)
            return self.lm_head(self.ln_f(x))

    return GPT2()


def _torch_gpt2(torch, nn, vocab, block, dim, depth, heads):
    class CausalSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = heads
            self.head_dim = dim // heads
            self.scale = self.head_dim ** -0.5
            self.c_attn = nn.Linear(dim, dim * 3)
            self.c_proj = nn.Linear(dim, dim)
            mask = torch.tril(torch.ones(block, block)).reshape(1, 1, block, block)
            self.register_buffer("mask", mask)

        def forward(self, x):
            b, t, c = x.shape
            qkv = self.c_attn(x).reshape(b, t, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            att = (q @ k.transpose(-2, -1)) * self.scale
            keep = self.mask[:, :, :t, :t]
            att = att * keep + (keep - 1.0) * 1e9
            att = att.softmax(dim=-1)
            y = (att @ v).transpose(1, 2).reshape(b, t, c)
            return self.c_proj(y)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_fc = nn.Linear(dim, dim * 4)
            self.c_proj = nn.Linear(dim * 4, dim)

        def forward(self, x):
            return self.c_proj(nn.functional.gelu(self.c_fc(x)))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_1 = nn.LayerNorm(dim)
            self.attn = CausalSelfAttention()
            self.ln_2 = nn.LayerNorm(dim)
            self.mlp = MLP()

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            return x + self.mlp(self.ln_2(x))

    class GPT2(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(vocab, dim)
            self.wpe = nn.Embedding(block, dim)
            self.h = nn.ModuleList([Block() for _ in range(depth)])
            self.ln_f = nn.LayerNorm(dim)
            self.lm_head = nn.Linear(dim, vocab, bias=False)

        def forward(self, idx):
            b, t = idx.shape
            pos = torch.arange(t, device=idx.device).expand(b, t)
            x = self.wte(idx) + self.wpe(pos)
            for layer in self.h:
                x = layer(x)
            return self.lm_head(self.ln_f(x))

    return GPT2()


# ------------------------------------------------------------------ diffusion

def _timestep_embedding_jittor(t, dim):
    half = dim // 2
    freqs = jt.exp(-math.log(10000.0) * jt.arange(half).float() / half)
    args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
    return jt.concat([jt.cos(args), jt.sin(args)], dim=-1)


def _jittor_unet(base, groups):
    class ResBlock(jnn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.norm1 = jnn.GroupNorm(groups, cin)
            self.conv1 = jnn.Conv2d(cin, cout, 3, padding=1)
            self.emb = jnn.Linear(base * 4, cout)
            self.norm2 = jnn.GroupNorm(groups, cout)
            self.conv2 = jnn.Conv2d(cout, cout, 3, padding=1)
            self.skip = jnn.Conv2d(cin, cout, 1) if cin != cout else jnn.Identity()

        def execute(self, x, emb):
            h = self.conv1(jnn.silu(self.norm1(x)))
            h = h + self.emb(jnn.silu(emb)).reshape(emb.shape[0], -1, 1, 1)
            h = self.conv2(jnn.silu(self.norm2(h)))
            return h + self.skip(x)

    class UNet(jnn.Module):
        def __init__(self):
            super().__init__()
            self.time_mlp1 = jnn.Linear(base, base * 4)
            self.time_mlp2 = jnn.Linear(base * 4, base * 4)
            self.conv_in = jnn.Conv2d(3, base, 3, padding=1)
            self.down = ResBlock(base, base * 2)
            self.pool = jnn.Pool(2, stride=2, op="maximum")
            self.mid = ResBlock(base * 2, base * 2)
            self.up = ResBlock(base * 4, base)
            self.norm_out = jnn.GroupNorm(groups, base)
            self.conv_out = jnn.Conv2d(base, 3, 3, padding=1)

        def execute(self, x, t):
            emb = self.time_mlp2(jnn.silu(self.time_mlp1(_timestep_embedding_jittor(t, base))))
            h0 = self.conv_in(x)
            h1 = self.down(h0, emb)
            h2 = self.mid(self.pool(h1), emb)
            h2 = jnn.interpolate(h2, size=(h1.shape[2], h1.shape[3]), mode="nearest")
            h = self.up(jt.concat([h1, h2], dim=1), emb)
            return self.conv_out(jnn.silu(self.norm_out(h)))

    return UNet()


def _torch_unet(torch, nn, base, groups):
    def timestep_embedding(t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    class ResBlock(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.norm1 = nn.GroupNorm(groups, cin)
            self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
            self.emb = nn.Linear(base * 4, cout)
            self.norm2 = nn.GroupNorm(groups, cout)
            self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
            self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x, emb):
            h = self.conv1(nn.functional.silu(self.norm1(x)))
            h = h + self.emb(nn.functional.silu(emb)).reshape(emb.shape[0], -1, 1, 1)
            h = self.conv2(nn.functional.silu(self.norm2(h)))
            return h + self.skip(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_mlp1 = nn.Linear(base, base * 4)
            self.time_mlp2 = nn.Linear(base * 4, base * 4)
            self.conv_in = nn.Conv2d(3, base, 3, padding=1)
            self.down = ResBlock(base, base * 2)
            self.pool = nn.MaxPool2d(2, stride=2)
            self.mid = ResBlock(base * 2, base * 2)
            self.up = ResBlock(base * 4, base)
            self.norm_out = nn.GroupNorm(groups, base)
            self.conv_out = nn.Conv2d(base, 3, 3, padding=1)

        def forward(self, x, t):
            emb = self.time_mlp2(nn.functional.silu(self.time_mlp1(timestep_embedding(t, base))))
            h0 = self.conv_in(x)
            h1 = self.down(h0, emb)
            h2 = self.mid(self.pool(h1), emb)
            h2 = nn.functional.interpolate(h2, size=(h1.shape[2], h1.shape[3]), mode="nearest")
            h = self.up(torch.cat([h1, h2], dim=1), emb)
            return self.conv_out(nn.functional.silu(self.norm_out(h)))

    return UNet()


# --------------------------------------------------------------------- ResNet

def _jittor_resnet():
    from jittor.models.resnet import Resnet18

    return Resnet18(num_classes=10)


def _torch_resnet(nn):
    from _helpers.torch_runtime import import_torch_modules

    (models,) = import_torch_modules("torchvision.models")
    return models.resnet18(num_classes=10)


# -------------------------------------------------------------------- registry

#: name -> (builder, needs_torchvision)
#:
#: A builder returns ``(jittor_module, torch_module, inputs)`` where ``inputs``
#: is a list of numpy arrays.  Integer arrays are passed through as token ids;
#: float arrays become differentiable inputs.
def build(name, seed=0):
    import numpy as np

    torch, nn, _f = _torch_modules()
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    if name == "resnet18":
        jittor_model = _jittor_resnet()
        torch_model = _torch_resnet(nn)
        inputs = [rng.randn(2, 3, 64, 64).astype("float32")]
    elif name == "vit":
        kw = dict(dim=64, depth=2, heads=4, patch=8, image=32, classes=10)
        jittor_model = _jittor_vit(
            kw["dim"], kw["depth"], kw["heads"], kw["patch"], kw["image"], kw["classes"]
        )
        torch_model = _torch_vit(
            torch, nn, kw["dim"], kw["depth"], kw["heads"], kw["patch"],
            kw["image"], kw["classes"],
        )
        inputs = [rng.randn(2, 3, 32, 32).astype("float32")]
    elif name == "gpt2":
        kw = dict(vocab=64, block=16, dim=64, depth=2, heads=4)
        jittor_model = _jittor_gpt2(**kw)
        torch_model = _torch_gpt2(torch, nn, **kw)
        inputs = [rng.randint(0, kw["vocab"], size=(2, 12)).astype("int32")]
    elif name == "diffusion_unet":
        jittor_model = _jittor_unet(base=16, groups=4)
        torch_model = _torch_unet(torch, nn, base=16, groups=4)
        inputs = [
            rng.randn(2, 3, 16, 16).astype("float32"),
            rng.randint(0, 100, size=(2,)).astype("int32"),
        ]
    else:
        raise KeyError(name)

    return jittor_model, torch_model, inputs


NETWORKS = ("resnet18", "vit", "gpt2", "diffusion_unet")
