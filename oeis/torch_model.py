
import torch, torch.nn as nn
from dataclasses import dataclass
from typing import List
import numpy as np

OPS = [
    "A","SCALE","OFFSET","MAP_ABS","MAP_SGN","MAP_MOD",
    "MAP_TAU","MAP_SIGMA","MAP_PHI","MAP_MU","MAP_OMEGA","MAP_BIGOMEGA",
    "SCAN_ADD","SCAN_MUL","CUMMAX","CUMMIN","DIFF_FWD","DIFF_BACK",
    "ZIP","CONV","POLY","REIDX","SUBSAMPLE","REPEAT",
    "INDEXBY","BINOM","IBINOM","EULER","DROP","INSERT1","INSERT2",
    "PRED_POS","PRED_NEG","PRED_IS_EVEN_N","PRED_EQ_CONST","PRED_GT_CONST","PRED_LT_CONST",
    "PRED_NOT","PRED_AND","PRED_OR","COND",
    "<BOS>","<EOS>"
]
INTS = list(range(-16,17))
BINOPS = ["ADD","SUB","MUL","MIN","MAX"]
TOKENS = OPS + BINOPS + [str(x) for x in INTS]
stoi = {t:i for i,t in enumerate(TOKENS)}
itos = {i:t for t,i in stoi.items()}

def cheap_features(A: List[int], B: List[int]):
    a = np.array(A, dtype=float); b = np.array(B, dtype=float)
    if len(a)==0: return torch.zeros(11, dtype=torch.float32)
    if np.all(a==a[0]):
        k=0.0; c=float(np.mean(b)) if len(b)>0 else 0.0
    else:
        A1 = np.vstack([a, np.ones(len(a))]).T
        k,c = np.linalg.lstsq(A1, b, rcond=None)[0]
    k = float(np.clip(k, -10, 10)); c = float(np.clip(c, -100, 100))
    db = np.zeros_like(b); 
    if len(b)>0: db[0]=b[0]
    if len(b)>1: db[1:] = np.diff(b)
    is_scan = 1.0 if len(b)>0 and np.allclose(db, a) else 0.0
    is_diff1 = 1.0 if len(b)>1 and np.allclose(b[1:], a[1:]-a[:-1]) and b[0]==0 else 0.0
    corr=[]
    for d in range(6):
        if d >= len(a) or d >= len(b) or (len(a)-d)<2: corr.append(0.0); continue
        va=a[d:]; vb=b[d:]
        sa=float(np.std(va)); sb=float(np.std(vb))
        if sa==0.0 or sb==0.0: corr.append(0.0)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                C = np.corrcoef(va, vb)[0,1]
            corr.append(0.0 if (np.isnan(C) or np.isinf(C)) else float(C))
    scale2 = 1.0 if np.allclose(b, 2*a) else 0.0
    return torch.tensor([k,c,is_scan,is_diff1,scale2]+corr, dtype=torch.float32)

@dataclass
class Cfg:
    d_model:int=256; nhead:int=4; nlayers:int=4; d_ff:int=1024; dropout:float=0.1; ctx_len:int=64; feat_dim:int=11

class TransDecoder(nn.Module):
    def __init__(self, cfg: Cfg, vocab:int=len(TOKENS)):
        super().__init__()
        self.cfg=cfg
        self.tok = nn.Embedding(vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.ctx_len, cfg.d_model)
        self.fproj = nn.Linear(cfg.feat_dim, cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.nlayers)
        self.lm = nn.Linear(cfg.d_model, vocab)
    def forward(self, ctx_ids: torch.LongTensor, feat: torch.FloatTensor):
        B,T = ctx_ids.shape
        T = min(T, self.cfg.ctx_len)
        ctx_ids = ctx_ids[:, -T:]
        pos = torch.arange(T, device=ctx_ids.device).unsqueeze(0).expand(B,T)
        x = self.tok(ctx_ids) + self.pos(pos) + self.fproj(feat).unsqueeze(1)
        mask = torch.triu(torch.ones(T,T,device=x.device, dtype=torch.bool), diagonal=1)
        h = self.enc(x, mask)
        return self.lm(h)
