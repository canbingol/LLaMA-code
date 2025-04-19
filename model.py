import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import math

@dataclass
class ModelArgs:
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads:  Optional[int] =6
    vocab_size: int = 128256
    multiple_of: int = 32
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 16
    max_seq_len: int = 246

    device: str = None

    train: bool = True
    
    batch_size: int  = 4
    drop_last: bool = True
    shuffle: bool = True


class RMSNorm(nn.Module):

    def __init__(self,dim, eps: float= 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.tensor):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.tensor):
        return self.gamma * self._norm(x)


def precompute_theta_freqs(seq_len, head_dim,device: str,theta: float= 10000.0):

    #! theta_i = 10000.0^(-2(x_i-1)/d) , x_i = [1,2,3, .. .... , d/2]

    # here we calculate 2(x_i-1) this part
    theta_numerator = torch.arange(0,head_dim,2).float()
    # 10000.0 ^ -2(theta / head_dim)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    m = torch.arange(seq_len, device=device)

    freqs = torch.outer(m,theta).float()

    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rope(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # x: (batch, seq_len, n_heads, head_dim)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # (B, S, H, D//2)
    
    # freqs_complex: (seq_len, head_dim//2) → reshape for broadcasting
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D//2)
    freqs_complex = freqs_complex.to(x_complex.device)

    assert freqs_complex.shape[1] == x_complex.shape[1], f"Seq len mismatch: {freqs_complex.shape[1]} vs {x_complex.shape[1]}"
    assert freqs_complex.shape[-1] == x_complex.shape[-1], f"Head dim mismatch: {freqs_complex.shape[-1]} vs {x_complex.shape[-1]}"
    x_rotated = x_complex * freqs_complex  # (B, S, H, D//2)
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)  # (B, S, H, D)
    return x_out.type_as(x).to(device)



def repeat_kv(x: torch.tensor, n_rep: int) -> torch.tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    
    return(
        # (batch_size, seq_len, n_kv_heads, 1, head_dim)
        x[:, :, :, None, :]

        # (batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .expand(batch_size, seq_len,n_kv_heads,n_rep,head_dim)

        #! (batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        .reshape(batch_size, seq_len,n_kv_heads * n_rep,head_dim)
    )

def apply_causal_mask(scores: torch.tensor, seq_len: int):

    mask = torch.triu(torch.ones(seq_len,seq_len), diagonal=1)
    mask = mask.to(scores.device)

    mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))

    return scores

class SelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.training = args.train # for controling KV-cache
        self.n_kv_heads = args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, self.n_heads_q * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
    
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        
    def forward(self, x: torch.tensor, freqs_comlex: torch.tensor, start_pos: int=None):
        
        batch_size, seq_len, _ = x.shape # (batch_size, seq_len, dim)
        #! (batch_size, seq_len,dim) -> = (batch_size, seq_len, head_dim * n_heads)
        xq = self.wq(x)

        #! (batch_size, seq_len,dim) -> = (batch_size, seq_len, n_kv_head_dim * n_kv_heads)
        xk = self.wk(x)
        xv = self.wv(x)
        
        #! (batch_size, seq_len, dim) -> (batch_size, seq_len, n_head_q, head_dim)
        xq = xq.view((batch_size, seq_len, self.n_heads_q, self.head_dim))

        #! (batch_size, seq_len, dim) -> (batch_size, seq_len, n_kv_head, head_dim)
        xk = xk.view((batch_size, seq_len, self.n_kv_heads, self.head_dim))
        xv = xv.view((batch_size, seq_len, self.n_kv_heads, self.head_dim))

        # applying RoPE
        xq = apply_rope(xq, freqs_comlex,device=x.device)
        xk = apply_rope(xk, freqs_comlex,device=x.device)

        if not self.training:
            self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
            self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

            keys = self.cache_k[:batch_size, :start_pos + seq_len]
            values = self.cache_v[:batch_size, :start_pos + seq_len]
        
        else:
            keys = xk
            values = xv

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        keys = keys.to(xq.device)
        values = values.to(xq.device)


        # (B, seq_len, H_Q, Head_Dim) -> (B, H_Q, seq_len, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = apply_causal_mask(scores,seq_len)

        scores = F.softmax(scores.float(),dim=-1).type_as(xq)   

        outout = torch.matmul(scores, values)
        # (B, H_Q, seq_len, Head_Dim) -> (B, seq_len, H_Q, Head_Dim) -> (B, seq_len, Dim)
        outout = (outout.transpose(1,2).contiguous().view(batch_size,seq_len,-1))
        return self.wo(outout)

class FFN(nn.Module):

    def __init__(self, args:ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    
    def forward(self, x: torch.tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x



class Encoder(nn.Module):

    def __init__(self, args:ModelArgs):
        super().__init__()
        dim = args.dim  

        self.attn = SelfAttention(args)
        self.ffn = FFN(args)
        self.norm_1 = RMSNorm(dim, args.norm_eps)
        self.norm_2 = RMSNorm(dim, args.norm_eps)

    def forward(self, x: torch.tensor ,freqs_comlex: torch.tensor, start_pos):

        h = x + self.attn.forward(
            self.norm_1(x), freqs_comlex, start_pos)

        out = h + self.ffn.forward(self.norm_2(h))
        return out


class LLaMA(nn.Module):

    def __init__(self, args:ModelArgs):
        super().__init__()
        assert args.vocab_size != -1, "Vocab size must be set"

        self.dim = args.dim
        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        self.args = args
        self.token_embd_table = nn.Embedding(self.vocab_size, self.dim)
        self.linear = nn.Linear(self.dim, self.vocab_size, bias=False)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(Encoder(args))

        self.norm = RMSNorm(self.dim, args.norm_eps)
        self.linear = nn.Linear(args.dim, self.vocab_size)

        self.freqs_complex = precompute_theta_freqs(
            seq_len=self.args.max_seq_len * 2,
            head_dim=self.args.dim // self.args.n_heads,
            device=self.args.device
        )

    def forward(self, x: torch.tensor, start_pos: int):
        batch_size, seq_len = x.shape
        x = x.long()
        h = self.token_embd_table(x)
        freqs_copmlex = self.freqs_complex[start_pos: start_pos+seq_len]

        for layer in self.layers:
            h = layer(h,freqs_copmlex, start_pos)
        h = self.norm(h).float()

        return self.linear(h)