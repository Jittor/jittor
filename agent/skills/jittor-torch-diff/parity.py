#!/usr/bin/env python
"""jittor-as-torch <-> real-torch differential parity harness.

Subcommands (run the SAME file under each python env):
  parity.py jt  <arch> <outdir>   # JITTOR env: build, save weights+input+grads
  parity.py rt  <arch> <outdir>   # REAL-TORCH env: load same weights, compute grads
  parity.py cmp <arch> <outdir>   # either env: compare forward + backward, print verdict

Compares forward (last_hidden_state) and per-parameter backward gradients, using
the network-wide-scaled grad metric (see SKILL.md). Loss = last_hidden_state
.float().pow(2).sum(). Tiny deterministic configs (dropout 0).
"""
import os, sys, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np


def make_config(arch):
    from transformers import AutoConfig
    if arch == "gpt2":
        return AutoConfig.for_model("gpt2", n_embd=64, n_layer=2, n_head=2,
            vocab_size=128, n_positions=64, n_inner=128,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    if arch == "bert":
        return AutoConfig.for_model("bert", hidden_size=64, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=128, vocab_size=128,
            max_position_embeddings=64, hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0)
    if arch == "llama":
        return AutoConfig.for_model("llama", hidden_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, intermediate_size=128,
            vocab_size=128, max_position_embeddings=64, attention_dropout=0.0)
    if arch == "gpt_neox":
        return AutoConfig.for_model("gpt_neox", hidden_size=64, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=128, vocab_size=128,
            max_position_embeddings=64)
    # Archs whose validators need explicit head_dim / expert / rotary_dim — the
    # generic builder below mis-sizes these (both jittor AND real torch reject the
    # config, so it's a harness artifact not a jittor bug). Give valid tiny dims.
    if arch == "qwen3":
        return AutoConfig.for_model("qwen3", hidden_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, head_dim=32,
            intermediate_size=128, vocab_size=128, max_position_embeddings=128,
            attention_dropout=0.0)
    if arch == "gemma":
        return AutoConfig.for_model("gemma", hidden_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=32,
            intermediate_size=128, vocab_size=128, max_position_embeddings=128,
            attention_dropout=0.0)
    if arch == "mixtral":
        return AutoConfig.for_model("mixtral", hidden_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, intermediate_size=128,
            num_local_experts=2, num_experts_per_tok=2, vocab_size=128,
            max_position_embeddings=128, attention_dropout=0.0)
    if arch == "gptj":
        return AutoConfig.for_model("gptj", n_embd=64, n_layer=2, n_head=2,
            rotary_dim=16, n_inner=128, vocab_size=128, n_positions=128,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    if arch == "resnet":
        return AutoConfig.for_model("resnet", num_channels=3, embedding_size=16,
            hidden_sizes=[16, 32], depths=[1, 1], layer_type="basic")
    if arch == "regnet":
        return AutoConfig.for_model("regnet", num_channels=3, embedding_size=16,
            hidden_sizes=[16, 32], depths=[1, 1], groups_width=8)
    if arch == "mobilenet_v2":
        return AutoConfig.for_model("mobilenet_v2", num_channels=3, image_size=32,
            depth_multiplier=0.5, expand_ratio=2)
    if arch in ("vit", "deit"):
        return AutoConfig.for_model(arch, hidden_size=64, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=128, image_size=32,
            patch_size=16, num_channels=3, hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0)
    if arch == "swin":
        return AutoConfig.for_model("swin", embed_dim=32, depths=[2, 2],
            num_heads=[2, 2], window_size=2, image_size=32, patch_size=4,
            num_channels=3, mlp_ratio=2.0, drop_path_rate=0.0,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    if arch == "convnext":
        return AutoConfig.for_model("convnext", hidden_sizes=[32, 64],
            depths=[1, 1], num_stages=2, image_size=32, num_channels=3,
            patch_size=4, drop_path_rate=0.0)
    if arch in ("falcon", "falconmha", "falconmq", "falconpar"):
        mq = arch in ("falcon", "falconmq")          # multi_query
        par = arch in ("falcon", "falconpar")        # parallel_attn
        return AutoConfig.for_model("falcon", hidden_size=64, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=128, vocab_size=128,
            max_position_embeddings=128, multi_query=mq, parallel_attn=par,
            new_decoder_architecture=False, bias=False, alibi=False,
            hidden_dropout=0.0, attention_dropout=0.0)
    if arch == "pegasus_x":
        return AutoConfig.for_model("pegasus_x", d_model=64, encoder_layers=2,
            decoder_layers=2, encoder_attention_heads=2, decoder_attention_heads=2,
            encoder_ffn_dim=128, decoder_ffn_dim=128, vocab_size=128,
            max_position_embeddings=512, dropout=0.0)
    if arch == "convbert":
        return AutoConfig.for_model("convbert", hidden_size=64, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=128, vocab_size=128,
            max_position_embeddings=128, head_ratio=2, conv_kernel_size=3, num_groups=1,
            embedding_size=64, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    if arch == "roformer":
        return AutoConfig.for_model("roformer", hidden_size=64, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=128, vocab_size=128,
            max_position_embeddings=128, hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0)
    if arch == "longformer":
        # sliding-window attention: small window, seq must be a multiple of 2*window
        return AutoConfig.for_model("longformer", hidden_size=64, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=128, vocab_size=128,
            max_position_embeddings=512, attention_window=4, pad_token_id=0,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    if arch == "gpt_neo":
        return AutoConfig.for_model("gpt_neo", hidden_size=64, num_layers=2,
            num_heads=2, attention_types=[[["global", "local"], 1]],
            intermediate_size=128, vocab_size=128, max_position_embeddings=128,
            resid_dropout=0.0, embed_dropout=0.0, attention_dropout=0.0)
    # Generic tiny config for any other arch: pass a broad superset of dim/dropout
    # aliases across transformers config classes (hidden_size/d_model/n_embd/dim,
    # num_hidden_layers/n_layer/num_layers/encoder_layers/..., etc.). Config
    # classes accept **kwargs and ignore the names they don't use, so each arch
    # picks up its own tiny dims and stays small + deterministic (dropout 0).
    g = dict(
        vocab_size=128, max_position_embeddings=128, pad_token_id=0,
        # width
        hidden_size=64, d_model=64, n_embd=64, dim=64, embed_dim=64, hidden_dim=64,
        # depth
        num_hidden_layers=2, n_layer=2, num_layers=2, n_layers=2,
        encoder_layers=2, decoder_layers=2,
        # heads
        num_attention_heads=2, n_head=2, num_heads=2, n_heads=2,
        encoder_attention_heads=2, decoder_attention_heads=2,
        num_key_value_heads=2,
        # ffn
        intermediate_size=128, n_inner=128, d_ff=128, ffn_dim=128,
        encoder_ffn_dim=128, decoder_ffn_dim=128, d_kv=16,
        # dropouts -> 0 for determinism
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        attention_dropout=0.0, dropout=0.0, resid_pdrop=0.0, embd_pdrop=0.0,
        attn_pdrop=0.0, activation_dropout=0.0,
    )
    return AutoConfig.for_model(arch, **g)


def fixed_ids():
    np.random.seed(0)
    return np.random.randint(0, 128, size=(2, 8)).astype(np.int64)


def build_inputs(model):
    """Return {kwarg: np_array} appropriate for the model type (deterministic).
    Text -> input_ids; encoder-decoder -> + decoder_input_ids; vision -> pixel_values.
    Saved to inputs.npz so the jittor and real-torch sides use identical inputs."""
    np.random.seed(0)
    cfg = model.config
    if getattr(cfg, "vocab_size", None):
        vocab = min(int(cfg.vocab_size), 128)
        inp = {"input_ids": np.random.randint(0, vocab, (2, 8)).astype(np.int64)}
        if getattr(cfg, "is_encoder_decoder", False):
            inp["decoder_input_ids"] = np.random.randint(0, vocab, (2, 8)).astype(np.int64)
        return inp
    # vision: pixel_values (B, C, H, W)
    c = int(getattr(cfg, "num_channels", 3)); h = int(getattr(cfg, "image_size", 32))
    return {"pixel_values": np.random.randn(2, c, h, h).astype(np.float32)}


def run_jt(arch, outdir):
    import torch, jittor as jt
    from transformers import AutoModel
    os.makedirs(outdir, exist_ok=True)
    model = AutoModel.from_config(make_config(arch)); model.eval()
    model.save_pretrained(outdir, safe_serialization=True)
    inp = build_inputs(model)
    np.savez(os.path.join(outdir, "inputs.npz"), **inp)
    named = list(model.named_parameters())            # registers leaves (torch_compat)
    out = model(**{k: torch.from_numpy(v) for k, v in inp.items()})
    hs = out.last_hidden_state
    loss = hs.float().pow(2).sum()
    # autodiff grads straight from jittor (ground truth on the jittor side)
    grads = jt.grad(loss, [p for _, p in named], retain_graph=True)
    np.savez(os.path.join(outdir, "grads_jt.npz"),
             **{n: np.asarray(g.float().numpy(), np.float32) for (n, _), g in zip(named, grads)})
    np.save(os.path.join(outdir, "hs_jt.npy"), np.asarray(hs.float().numpy(), np.float32))
    json.dump({"loss": float(np.asarray(loss.float().numpy()).reshape(-1)[0]),
               "nparams": len(named),
               "ids_sum": int(sum(int(v.sum()) for v in inp.values() if v.dtype.kind in "iu"))},
              open(os.path.join(outdir, "meta_jt.json"), "w"))
    print("JT", arch, "loss=", float(np.asarray(loss.float().numpy()).reshape(-1)[0]),
          "nparams=", len(named))


def run_rt(arch, outdir):
    import torch
    from transformers import AutoModel
    assert not hasattr(torch, "jittor"), "RT env is not real torch!"
    res = AutoModel.from_pretrained(outdir, output_loading_info=True)
    model, info = res if isinstance(res, tuple) else (res, {})
    # num_batches_tracked is a non-numeric BN counter that torch stores as a 0-d
    # scalar; jittor has no 0-d tensors so it can't shape-match, and keeps it
    # non-persistent -> torch reports it "missing". It doesn't affect forward/
    # backward numerics, so exclude it from the key-integrity check.
    _ignore = lambda k: k.endswith("num_batches_tracked")
    missing = [k for k in info.get("missing_keys", []) if not _ignore(k)]
    unexpected = [k for k in info.get("unexpected_keys", []) if not _ignore(k)]
    model.eval()
    _d = np.load(os.path.join(outdir, "inputs.npz")); inp = {k: _d[k] for k in _d.files}
    out = model(**{k: torch.from_numpy(v) for k, v in inp.items()})
    hs = out.last_hidden_state
    loss = hs.float().pow(2).sum(); loss.backward()
    grads = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            grads[n] = p.grad.float().detach().numpy().astype(np.float32)
    np.savez(os.path.join(outdir, "grads_rt.npz"), **grads)
    np.save(os.path.join(outdir, "hs_rt.npy"), hs.float().detach().numpy().astype(np.float32))
    json.dump({"loss": float(loss.detach().float().numpy()), "torch": torch.__version__,
               "missing": missing, "unexpected": unexpected, "nparams": len(grads)},
              open(os.path.join(outdir, "meta_rt.json"), "w"))
    print("RT", arch, "loss=", float(loss.detach().float().numpy()),
          "torch=", torch.__version__, "missing=", len(missing), "unexpected=", len(unexpected))


def run_cmp(arch, outdir):
    gj = np.load(os.path.join(outdir, "grads_jt.npz"))
    gr = np.load(os.path.join(outdir, "grads_rt.npz"))
    mj = json.load(open(os.path.join(outdir, "meta_jt.json")))
    mr = json.load(open(os.path.join(outdir, "meta_rt.json")))
    hj = np.load(os.path.join(outdir, "hs_jt.npy")).astype(np.float64)
    hr = np.load(os.path.join(outdir, "hs_rt.npy")).astype(np.float64)
    fwd = float(np.max(np.abs(hj - hr)) / (np.max(np.abs(hr)) + 1e-8))
    loss_rel = abs(mj["loss"] - mr["loss"]) / (abs(mr["loss"]) + 1e-8)
    common = [k for k in gr.files if k in gj.files]
    gmax = max((float(np.max(np.abs(gr[k]))) for k in common), default=1.0) + 1e-8
    worst = 0.0; worstn = None
    for k in common:
        a = gj[k].astype(np.float64); b = gr[k].astype(np.float64)
        if a.shape != b.shape:
            print("  SHAPE MISMATCH", k, a.shape, b.shape); continue
        d = float(np.max(np.abs(a - b)) / gmax)
        if d > worst: worst, worstn = d, k
    only_rt = sorted(set(gr.files) - set(gj.files))
    only_jt = sorted(set(gj.files) - set(gr.files))
    fwd_ok = fwd < 1e-4; bwd_ok = worst < 1e-5; key_ok = not mr["missing"] and not mr["unexpected"]
    print(f"[{arch}] torch={mr.get('torch')}  load: missing={len(mr['missing'])} unexpected={len(mr['unexpected'])}")
    print(f"  FORWARD  rel_diff(last_hidden_state) = {fwd:.2e}   loss_rel = {loss_rel:.2e}   {'PASS' if fwd_ok else 'FAIL'}")
    print(f"  BACKWARD net-scaled worst grad rel   = {worst:.2e} @ {worstn}   {'PASS' if bwd_ok else 'FAIL'}")
    if only_rt: print(f"  grads only in RT (jittor missing): {only_rt}")
    if only_jt: print(f"  grads only in JT: {only_jt}")
    print(f"  VERDICT: {'PASS' if (fwd_ok and bwd_ok and key_ok) else 'FAIL'}")


if __name__ == "__main__":
    cmd, arch, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    {"jt": run_jt, "rt": run_rt, "cmp": run_cmp}[cmd](arch, outdir)
