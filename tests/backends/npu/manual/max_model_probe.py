"""Find the largest GPT-style model trainable on ONE 910B3 (64GB), then
report it as the max for 8-card data-parallel (DP replicates the model, so the
per-card memory ceiling is what bounds model size; 8 cards give ~8x throughput).
"""
import sys, time, numpy as np, jittor as jt
# NB: use_acl alone routes to native ACL ops but leaves use_cuda=0, which keeps
# *execution on CPU* (measuring host RAM, not the 64GB HBM). Must set use_cuda=1
# so allocations actually land on the NPU.
jt.flags.use_acl = 1
jt.flags.use_cuda = 1
from jittor import nn
import math

DTYPE = "float32"   # over/set below from argv; bf16 ~halves weight+grad memory

class RMSNorm(nn.Module):
    # modern LLMs (Qwen3/Llama) use RMSNorm; its ACL path is bf16-safe whereas
    # the native LayerNorm op currently errors on bf16 (161002).
    def __init__(s, E, eps=1e-6):
        s.weight = jt.ones((E,)); s.eps = eps
    def execute(s, x):
        v = (x.float32() * x.float32()).mean(-1, keepdims=True)
        return (x.float32() * jt.rsqrt(v + s.eps)).cast(x.dtype) * s.weight

def make_gpt(L, H, E, V=50257, T=1024):
    class Blk(nn.Module):
        def __init__(s):
            s.ln1=RMSNorm(E); s.ln2=RMSNorm(E)
            s.qkv=nn.Linear(E,3*E); s.proj=nn.Linear(E,E)
            s.fc=nn.Linear(E,4*E); s.fc2=nn.Linear(4*E,E); s.nh=H
        def execute(s,x):
            B,Tt,C=x.shape
            qkv=s.qkv(s.ln1(x)); q,k,v=qkv.chunk(3,dim=2)
            q=q.reshape(B,Tt,s.nh,C//s.nh).transpose(0,2,1,3)
            k=k.reshape(B,Tt,s.nh,C//s.nh).transpose(0,2,1,3)
            v=v.reshape(B,Tt,s.nh,C//s.nh).transpose(0,2,1,3)
            a=nn.softmax(jt.matmul(q,k.transpose(0,1,3,2))/math.sqrt(C//s.nh),dim=-1)
            y=jt.matmul(a,v).transpose(0,2,1,3).reshape(B,Tt,C)
            x=x+s.proj(y); x=x+s.fc2(nn.gelu(s.fc(s.ln2(x))))
            return x
    class GPT(nn.Module):
        def __init__(s):
            s.wte=nn.Embedding(V,E); s.wpe=nn.Embedding(T,E)
            s.blocks=nn.ModuleList([Blk() for _ in range(L)])
            s.lnf=RMSNorm(E); s.head=nn.Linear(E,V,bias=False)
        def execute(s,idx,tgt):
            B,Tt=idx.shape
            x=s.wte(idx)+s.wpe(jt.arange(Tt).reshape(1,Tt))
            for b in s.blocks: x=b(x)
            # loss in fp32 (standard mixed-precision practice; bf16 cross-entropy
            # over a 50k vocab is numerically poor and hits ACL op gaps).
            logits = s.head(s.lnf(x)).reshape(-1,V).float32()
            return nn.cross_entropy_loss(logits, tgt.reshape(-1))
    return GPT()

def try_train(L,H,E,B,T,dtype="float32",lora=False):
    m=make_gpt(L,H,E,T=T)
    if dtype!="float32":
        for p in m.parameters():
            p.assign(p.cast(dtype))
    if lora:
        # LoRA-style: freeze base, train only a small adapter -> no optimizer
        # state for the frozen bulk (this is what bounds large-model LoRA).
        for p in m.parameters():
            p.stop_grad()
        adapter=[jt.zeros((E,16)).cast(dtype), jt.zeros((16,E)).cast(dtype)]
        for a in adapter: a.start_grad()
        opt=jt.optim.Adam(adapter,lr=1e-4)
    else:
        opt=jt.optim.Adam(m.parameters(),lr=1e-4)
    nparams=sum(p.numel() for p in m.parameters())
    d=np.random.randint(0,50257,size=(B,T+1)).astype(np.int32)
    x=jt.array(d[:,:-1]); y=jt.array(d[:,1:])
    loss=m(x,y)
    if lora:
        loss=loss + 0.0*(x.float32().sum()*0)  # keep graph; adapter unused in toy fwd
    opt.step(loss); jt.sync_all(True)
    return nparams

if __name__=="__main__":
    L,H,E,B,T = [int(a) for a in sys.argv[1:6]]
    dtype = sys.argv[6] if len(sys.argv)>6 else "float32"
    try:
        n=try_train(L,H,E,B,T,dtype=dtype)
        print(f"OK L={L} H={H} E={E} B={B} T={T} dt={dtype} params={n/1e6:.0f}M", flush=True)
    except Exception as e:
        print(f"OOM/ERR L={L} H={H} E={E} B={B} T={T} dt={dtype}: {repr(e)[:80]}", flush=True)
