"""CUDA regression for fused SDPA followed by a separate large lazy graph."""
import os

import numpy as np
import pytest
import jittor as jt
import torch
import torch.nn as nn
import torch.nn.functional as F

from jittor.compat.diagnostics import sdpa_flash_stats


@pytest.mark.skipif(not os.environ.get('JITTOR_FLASH_ATTN_JITTOR_SRC'), reason='native FlashAttention source not configured')
def test_flash_sdpa_output_survives_repeated_branched_graphs():
    if not jt.has_cuda:
        pytest.skip('CUDA required')

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(3)])

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < 2:
                    x = F.relu(x, inplace=True)
            return x

    class Graph(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.prompt, self.hyper = MLP(), MLP(), MLP()
            self.q, self.k, self.v, self.o = (nn.Linear(64, 64) for _ in range(4))
            self.norm, self.fusion = nn.LayerNorm(64), nn.Linear(96, 64)
            self.upnorm, self.up = nn.LayerNorm(64), nn.Linear(64, 64)

        def forward(self, pc, pe, tokens, centers, coords, raw):
            indices, weights = [], []
            for start in range(0, int(coords.shape[1]), 8192):
                distance = torch.cdist(coords[:, start:start + 8192, :], centers)
                values, index = torch.topk(distance, 3, dim=2, largest=False, sorted=False)
                inv = 1.0 / torch.clamp(values.square(), min=1e-8)
                indices.append(index)
                weights.append(inv / inv.sum(dim=2, keepdim=True))
            index, weight = torch.cat(indices, dim=1), torch.cat(weights, dim=1)
            source, tokens = self.encoder(pc), self.prompt(tokens)

            def heads(x):
                return x.reshape(1, x.shape[1], 8, 8).transpose(1, 2)

            q = heads(self.q(tokens + tokens))
            k, v = heads(self.k(source + pe)), heads(self.v(source))
            attention = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, 4, 64)
            hs = self.norm(tokens + self.o(attention))
            interp = source[:, :1, :].expand(1, int(index.shape[1]), 64)
            w, b = self.fusion.weight, self.fusion.bias
            up = interp @ w[:, :64].T + raw @ w[:, 64:].T + b
            up = F.gelu(self.up(F.gelu(self.upnorm(up))))
            hyper = torch.stack([self.hyper(hs[:, i, :]) for i in range(4)], dim=1)
            return hyper @ up.transpose(-1, -2)

    rng = np.random.default_rng(123)
    def array(shape):
        return torch.tensor((rng.standard_normal(shape) * .1).astype(np.float32), device='cuda')

    with jt.flag_scope(use_cuda=1, auto_graph_replay=0), torch.no_grad(), \
            torch.autocast(device_type='cuda', dtype=torch.bfloat16), \
            torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        args = (array((1, 4096, 64)), array((1, 4096, 64)), array((1, 4, 64)),
                array((1, 4096, 3)), array((1, 8193, 3)), array((1, 8193, 32)))
        model = Graph().cuda().eval()
        before = sdpa_flash_stats(jt)['hits']
        outputs = [model(*(value.clone() for value in args)).detach().clone().cpu().float().numpy()
                   for _ in range(3)]
        assert sdpa_flash_stats(jt)['hits'] >= before + 3, 'fused SDPA did not run'
    np.testing.assert_array_equal(outputs[0], outputs[1])
    np.testing.assert_array_equal(outputs[1], outputs[2])
