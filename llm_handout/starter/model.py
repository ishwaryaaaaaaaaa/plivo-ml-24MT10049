"""A small GPT in plain PyTorch. Yours to modify or replace entirely —
attention, SSM, whatever — as long as evaluate.py still works and the
parameter cap holds.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 256      # byte-level tokenizer default
    block_size = 256
    n_layer = 4
    n_head = 4
    n_embd = 176
    dropout = 0.0
    tie_weights = True    # share tok_emb <-> head weights (saves ~1 embed table)
    pos_encoding = "rope"  # "learned" (absolute nn.Embedding) or "rope"
    init_std = 0.02        # GPT-2 style; residual-stream projections scaled further


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_dim); cos/sin: (T, head_dim)
    return x * cos[None, None, :, :] + rotate_half(x) * sin[None, None, :, :]


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.pos_encoding = cfg.pos_encoding
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.pos_encoding == "rope" and rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos[:T], sin[:T])
            k = apply_rope(k, cos[:T], sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x, rope=None):
        x = x + self.attn(self.ln1(x), rope=rope)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.use_rope = getattr(cfg, "pos_encoding", "learned") == "rope"
        if self.use_rope:
            head_dim = cfg.n_embd // cfg.n_head
            inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
            t = torch.arange(cfg.block_size).float()
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.register_buffer("rope_cos", emb.cos(), persistent=False)
            self.register_buffer("rope_sin", emb.sin(), persistent=False)
        else:
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)
        # GPT-2 style: scale down residual-stream projections by 1/sqrt(2*n_layer)
        # so residual variance doesn't grow with depth (Radford et al. 2019 init).
        std = getattr(cfg, "init_std", 0.02) / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=std)

    def _init(self, m):
        std = getattr(self.cfg, "init_std", 0.02)
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if self.use_rope:
            x = self.drop(self.tok_emb(idx))
            rope = (self.rope_cos, self.rope_sin)
        else:
            pos = torch.arange(T, device=idx.device)
            x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
            rope = None
        for blk in self.blocks:
            x = blk(x, rope=rope)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
