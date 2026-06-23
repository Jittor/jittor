# ***************************************************************
# torch-level regression test (#6): run real transformers models
# through `import torch` -> jittor, on the torch-compat layer.
#
# REQUIRES an env with the torch_shim deployed + transformers, e.g. the
# py3.11 conda env used for jittor-as-torch:
#   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1
#   /home/yizhang/miniconda3/envs/jt-torch/bin/python -m jittor.test.test_torch_hf_models
# Skips cleanly if torch_shim/transformers are unavailable.
#
# Covers ~30 architectures (decoder / encoder / encoder-decoder / vision):
# forward finiteness + eval-determinism (Dropout off), param.grad populated after
# loss.backward() (the no-optimizer autograd bridge), buffer/param separation,
# recursive named_buffers, torch.norm scalar reduction, and Var.T. The configs
# mirror the validated jittor-torch-diff parity set (all PASS vs real torch ~1e-6).
# ***************************************************************
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DEACTIVATE_ASYNC_LOAD', '1')
import unittest, numpy as np

try:
    import torch  # torch_shim -> jittor
    import jittor as jt
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    _HAS = (getattr(torch, '__name__', '') == 'torch') and hasattr(torch, 'tensor')
except Exception:
    _HAS = False

CFG = {
 # decoder LLMs
 'gpt2':  dict(n_layer=2,n_embd=64,n_head=2,vocab_size=128,n_positions=64,resid_pdrop=0.5,embd_pdrop=0.5,attn_pdrop=0.5),
 'llama': dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=128,max_position_embeddings=64),
 'qwen2': dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=128,max_position_embeddings=64),
 'qwen3': dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,head_dim=32,vocab_size=128,max_position_embeddings=128),
 'mistral':   dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=128,max_position_embeddings=64),
 'gemma':     dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=32,vocab_size=128,max_position_embeddings=64),
 'gemma2':    dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=32,vocab_size=128,max_position_embeddings=128),
 'phi':       dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,vocab_size=128,max_position_embeddings=64),
 'phi3':      dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=128,max_position_embeddings=128),
 'opt':   dict(hidden_size=64,ffn_dim=128,num_hidden_layers=2,num_attention_heads=2,vocab_size=128,max_position_embeddings=64,word_embed_proj_dim=64,dropout=0.5),
 'bloom': dict(hidden_size=64,n_layer=2,n_head=2,vocab_size=128,hidden_dropout=0.5,attention_dropout=0.5),
 'gpt_neox':  dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,vocab_size=128,max_position_embeddings=64),
 'gptj':      dict(n_embd=64,n_layer=2,n_head=2,rotary_dim=16,n_inner=128,vocab_size=128,n_positions=128),
 'gpt_neo':   dict(hidden_size=64,num_layers=2,num_heads=2,attention_types=[[["global","local"],1]],intermediate_size=128,vocab_size=128,max_position_embeddings=128),
 'stablelm':  dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=128,max_position_embeddings=128),
 'starcoder2':dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=128,max_position_embeddings=128),
 'mpt':       dict(d_model=64,n_heads=2,n_layers=2,vocab_size=128,max_seq_len=128,expansion_ratio=2),
 # encoder
 'bert':  dict(hidden_size=64,num_hidden_layers=2,num_attention_heads=2,intermediate_size=128,vocab_size=128,max_position_embeddings=64,hidden_dropout_prob=0.5,attention_probs_dropout_prob=0.5),
 'roberta':   dict(hidden_size=64,num_hidden_layers=2,num_attention_heads=2,intermediate_size=128,vocab_size=128,max_position_embeddings=64,hidden_dropout_prob=0.5),
 'electra':   dict(hidden_size=64,num_hidden_layers=2,num_attention_heads=2,intermediate_size=128,vocab_size=128,max_position_embeddings=64,embedding_size=64,hidden_dropout_prob=0.5),
 'albert':    dict(hidden_size=64,num_hidden_layers=2,num_attention_heads=2,intermediate_size=128,vocab_size=128,max_position_embeddings=64,embedding_size=64),
 'distilbert':dict(dim=64,hidden_dim=128,n_layers=2,n_heads=2,vocab_size=128,max_position_embeddings=64),
 'longformer':dict(hidden_size=64,num_hidden_layers=2,num_attention_heads=2,intermediate_size=128,vocab_size=128,max_position_embeddings=512,attention_window=4,hidden_dropout_prob=0.5,attention_probs_dropout_prob=0.5),  # sliding-window attn (as_strided/where/flip)
 # encoder-decoder
 't5':    dict(d_model=64,d_ff=128,num_layers=2,num_heads=2,d_kv=32,vocab_size=128,dropout_rate=0.5),
 'bart':  dict(d_model=64,encoder_layers=2,decoder_layers=2,encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=128,decoder_ffn_dim=128,vocab_size=128,max_position_embeddings=128,dropout=0.0),
 'mbart': dict(d_model=64,encoder_layers=2,decoder_layers=2,encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=128,decoder_ffn_dim=128,vocab_size=128,max_position_embeddings=128,dropout=0.0),
 'pegasus':dict(d_model=64,encoder_layers=2,decoder_layers=2,encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=128,decoder_ffn_dim=128,vocab_size=128,max_position_embeddings=128,dropout=0.0),
 # vision
 'vit':   dict(hidden_size=64,num_hidden_layers=2,num_attention_heads=2,intermediate_size=128,image_size=32,patch_size=16,num_channels=3),
 'deit':  dict(hidden_size=64,num_hidden_layers=2,num_attention_heads=2,intermediate_size=128,image_size=32,patch_size=16,num_channels=3),
 'swin':  dict(embed_dim=32,depths=[2,2],num_heads=[2,2],window_size=2,image_size=32,patch_size=4,num_channels=3,mlp_ratio=2.0),
 'convnext':dict(hidden_sizes=[32,64],depths=[1,1],num_stages=2,image_size=32,num_channels=3,patch_size=4),
}


def _build(a):
    # pad_token_id=0 keeps the padding index inside the tiny vocab (some configs,
    # e.g. phi3, default pad_token_id to a real-vocab value like 32000).
    cfg = dict(CFG[a]); cfg.setdefault('pad_token_id', 0)
    return AutoModel.from_config(AutoConfig.for_model(a, **cfg))


def _inp(m):
    """Model-type-aware inputs: text -> input_ids; enc-dec -> +decoder_input_ids;
    vision -> pixel_values."""
    cfg = m.config
    if getattr(cfg, 'vocab_size', None):
        ids = torch.tensor(np.random.randint(0, min(int(cfg.vocab_size), 128), (1, 8)).astype('int64'))
        d = dict(input_ids=ids)
        if getattr(cfg, 'is_encoder_decoder', False):
            d['decoder_input_ids'] = ids
        return d
    c = int(getattr(cfg, 'num_channels', 3)); h = int(getattr(cfg, 'image_size', 32))
    return dict(pixel_values=torch.tensor(np.random.randn(1, c, h, h).astype('float32')))


@unittest.skipUnless(_HAS, "needs torch_shim + transformers")
class TestTorchHFModels(unittest.TestCase):
    def test_forward_and_eval_determinism(self):
        for a, cfg in CFG.items():
            with self.subTest(model=a):
                m = _build(a); m.eval()
                x = _inp(m)
                with torch.no_grad():
                    h1 = m(**x).last_hidden_state.float().numpy()
                    h2 = m(**x).last_hidden_state.float().numpy()
                self.assertTrue(np.isfinite(h1).all(), f"{a} non-finite")
                self.assertTrue(np.allclose(h1, h2, atol=1e-5), f"{a} eval non-deterministic (Dropout active?)")

    def test_grad_populated_after_backward(self):
        # Regression for the no-optimizer autograd bridge: enumerating params then
        # loss.backward() must populate param.grad for every trainable param.
        for a in ('gpt2', 'llama', 'bert', 't5', 'vit', 'bloom', 'falcon', 'mpt'):
            if a not in CFG:
                continue
            with self.subTest(model=a):
                m = _build(a); m.eval()
                named = list(m.named_parameters())
                loss = m(**_inp(m)).last_hidden_state.float().pow(2).sum()
                loss.backward()
                none = [n for n, p in named if p.grad is None]
                self.assertEqual(none, [], f"{a}: {len(none)} params have None grad after backward")
                m.zero_grad()
                self.assertTrue(all(p.grad is None for _, p in named), f"{a}: zero_grad did not clear")

    def test_buffers_not_in_parameters(self):
        m = AutoModelForCausalLM.from_config(AutoConfig.for_model('llama', **CFG['llama']))
        self.assertEqual(len(list(m.parameters())), len(list(m.named_parameters())))
        nb = [n for n, _ in m.named_buffers()]
        self.assertTrue(any('inv_freq' in n for n in nb), "named_buffers not recursing")
        self.assertFalse(any('inv_freq' in n for n in [n for n, _ in m.named_parameters()]), "buffer leaked into params")

    def test_norm_scalar_and_T(self):
        x = torch.tensor(np.random.randn(3, 4).astype('float32'))
        self.assertAlmostEqual(float(torch.norm(x)), float(np.linalg.norm(x.float().numpy())), places=3)
        self.assertEqual(tuple(x.T.shape), (4, 3))


if __name__ == '__main__':
    unittest.main()
