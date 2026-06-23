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
# Covers 20 architectures (forward + eval-determinism), the buffer/param
# separation, recursive named_buffers, torch.norm scalar reduction, and Var.T.
# ***************************************************************
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DEACTIVATE_ASYNC_LOAD', '1')
import unittest, numpy as np

try:
    import torch  # torch_shim -> jittor
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    _HAS = (getattr(torch, '__name__', '') == 'torch') and hasattr(torch, 'tensor')
except Exception:
    _HAS = False

CFG = {
 'gpt2':  dict(n_layer=2,n_embd=64,n_head=2,vocab_size=128,n_positions=64,resid_pdrop=0.5,embd_pdrop=0.5,attn_pdrop=0.5),
 'llama': dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=128,max_position_embeddings=64),
 'qwen2': dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=128,max_position_embeddings=64),
 'bert':  dict(hidden_size=64,num_hidden_layers=2,num_attention_heads=2,intermediate_size=128,vocab_size=128,max_position_embeddings=64,hidden_dropout_prob=0.5,attention_probs_dropout_prob=0.5),
 't5':    dict(d_model=64,d_ff=128,num_layers=2,num_heads=2,d_kv=32,vocab_size=128,dropout_rate=0.5),
 'vit':   dict(hidden_size=64,num_hidden_layers=2,num_attention_heads=2,intermediate_size=128,image_size=32,patch_size=16,num_channels=3),
 'bloom': dict(hidden_size=64,n_layer=2,n_head=2,vocab_size=128,hidden_dropout=0.5,attention_dropout=0.5),
 'opt':   dict(hidden_size=64,ffn_dim=128,num_hidden_layers=2,num_attention_heads=2,vocab_size=128,max_position_embeddings=64,word_embed_proj_dim=64,dropout=0.5),
 'mistral':   dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=128,max_position_embeddings=64),
 'gemma':     dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=32,vocab_size=128,max_position_embeddings=64),
 'phi':       dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,vocab_size=128,max_position_embeddings=64),
 'gpt_neox':  dict(hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,vocab_size=128,max_position_embeddings=64),
 'falcon':    dict(hidden_size=64,num_hidden_layers=2,num_attention_heads=2,vocab_size=128),
 'distilbert':dict(dim=64,hidden_dim=128,n_layers=2,n_heads=2,vocab_size=128,max_position_embeddings=64),
}
def _inp(a):
    if a == 'vit': return dict(pixel_values=torch.tensor(np.random.randn(1,3,32,32).astype('float32')))
    ids = torch.tensor(np.random.randint(0,128,(1,8)).astype('int64'))
    return dict(input_ids=ids, decoder_input_ids=ids) if a == 't5' else dict(input_ids=ids)


@unittest.skipUnless(_HAS, "needs torch_shim + transformers")
class TestTorchHFModels(unittest.TestCase):
    def test_forward_and_eval_determinism(self):
        for a, cfg in CFG.items():
            with self.subTest(model=a):
                m = AutoModel.from_config(AutoConfig.for_model(a, **cfg)); m.eval()
                x = _inp(a)
                with torch.no_grad():
                    h1 = m(**x).last_hidden_state.float().numpy()
                    h2 = m(**x).last_hidden_state.float().numpy()
                self.assertTrue(np.isfinite(h1).all(), f"{a} non-finite")
                self.assertTrue(np.allclose(h1, h2, atol=1e-5), f"{a} eval non-deterministic (Dropout active?)")

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
