"""Find the largest GPT-style model trainable on ONE 910B3 (64GB), then
report it as the max for 8-card data-parallel (DP replicates the model, so the
per-card memory ceiling is what bounds model size; 8 cards give ~8x throughput).
"""
import sys, time, numpy as np, jittor as jt
jt.flags.use_acl = 1
from jittor import nn
import math

def make_gpt(L, H, E, V=50257, T=1024):
    class Blk(nn.Module):
        def __init__(s):
            s.ln1=nn.LayerNorm(E); s.ln2=nn.LayerNorm(E)
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
            s.lnf=nn.LayerNorm(E); s.head=nn.Linear(E,V,bias=False)
        def execute(s,idx,tgt):
            B,Tt=idx.shape
            x=s.wte(idx)+s.wpe(jt.arange(Tt).reshape(1,Tt))
            for b in s.blocks: x=b(x)
            return nn.cross_entropy_loss(s.head(s.lnf(x)).reshape(-1,V), tgt.reshape(-1))
    return GPT()

def try_train(L,H,E,B,T):
    m=make_gpt(L,H,E,T=T); opt=jt.optim.Adam(m.parameters(),lr=1e-4)
    nparams=sum(p.numel() for p in m.parameters())
    d=np.random.randint(0,50257,size=(B,T+1)).astype(np.int32)
    x=jt.array(d[:,:-1]); y=jt.array(d[:,1:])
    opt.step(m(x,y)); jt.sync_all(True)
    return nparams

if __name__=="__main__":
    L,H,E,B,T = [int(a) for a in sys.argv[1:6]]
    try:
        n=try_train(L,H,E,B,T)
        print(f"OK L={L} H={H} E={E} B={B} T={T} params={n/1e6:.0f}M", flush=True)
    except Exception as e:
        print(f"OOM/ERR L={L} H={H} E={E} B={B} T={T}: {repr(e)[:80]}", flush=True)
