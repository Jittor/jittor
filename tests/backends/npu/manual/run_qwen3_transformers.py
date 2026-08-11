import torch, jittor as jt
jt.flags.use_acl = 1
from transformers import AutoModelForCausalLM, AutoTokenizer
mp = "/home/yizhang/models/Qwen3-0.6B"
print("loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(mp)
print("loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(mp, dtype="float32")
print("LOADED params:", sum(p.numel() for p in model.parameters())/1e6, "M", flush=True)
model.eval()
msgs = [{"role":"user","content":"Hello, what is 2+2?"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="np")["input_ids"]
import numpy as np
ids = torch.from_numpy(ids.astype(np.int64))
with torch.no_grad():
    gen = model.generate(ids, max_new_tokens=20, do_sample=False)
out = tok.decode(gen.numpy()[0][ids.shape[1]:], skip_special_tokens=True)
print("GEN:", repr(out), flush=True)
print("REAL_QWEN_OK", flush=True)
