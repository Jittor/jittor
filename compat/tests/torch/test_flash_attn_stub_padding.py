import torch

from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import pad_input, unpad_input


def test_flash_attn_padding_round_trip():
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 1]], dtype=torch.int32)

    packed, indices, cu_seqlens, max_seqlen, used = unpad_input(hidden, mask)
    restored = pad_input(packed, indices, batch=2, seqlen=4)

    expected = hidden * mask.unsqueeze(-1)
    assert max_seqlen == 3
    assert used.tolist() == [2, 3]
    assert cu_seqlens.tolist() == [0, 2, 5]
    torch.testing.assert_close(restored, expected)


def test_flash_attn_varlen_gqa_expands_kv_heads():
    q = torch.randn(4, 4, 8, requires_grad=True)
    k = torch.randn(4, 2, 8, requires_grad=True)
    v = torch.randn(4, 2, 8, requires_grad=True)
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)

    result = flash_attn_varlen_func(q, k, v, cu, cu, 2, 2, causal=True)
    expected = []
    for start in (0, 2):
        q_part = q[start:start + 2].permute(1, 0, 2).unsqueeze(0)
        k_part = k[start:start + 2].repeat_interleave(2, dim=1).permute(1, 0, 2).unsqueeze(0)
        v_part = v[start:start + 2].repeat_interleave(2, dim=1).permute(1, 0, 2).unsqueeze(0)
        expected.append(torch.nn.functional.scaled_dot_product_attention(q_part, k_part, v_part, is_causal=True)
                        .squeeze(0).permute(1, 0, 2))
    torch.testing.assert_close(result, torch.cat(expected, dim=0))
    result.sum().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None
