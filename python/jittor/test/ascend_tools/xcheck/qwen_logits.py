"""Dump Qwen3-0.6B next-token logits for a fixed prompt. Runs under real torch OR
the jittor shim (import torch resolves to whichever env). fp32."""
import sys, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
PATH="/home/yizhang/models/Qwen3-0.6B"; tag=sys.argv[1] if len(sys.argv)>1 else "x"
tok=AutoTokenizer.from_pretrained(PATH)
model=AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.float32); model.eval()
ids=tok("The capital of France is", return_tensors="pt")["input_ids"]
out=model(input_ids=ids)
last=out.logits[0,-1]
arr=last.detach().float().numpy() if hasattr(last,"detach") else np.asarray(last.numpy()).astype("float64")
arr=arr.astype("float64")
top5=arr.argsort()[-5:][::-1].tolist()
res={"tag":tag,"logit_l2":float((arr**2).sum()**0.5),"argmax":int(arr.argmax()),
     "top5":top5,"top5_logits":[float(arr[i]) for i in top5],"first8":[float(x) for x in arr[:8]]}
json.dump(res, open(f"/home/yizhang/xcheck/qwen_{tag}.json","w"), indent=2)
print(tag, "argmax", res["argmax"], "top5", top5, "logit_l2 %.4f"%res["logit_l2"])
