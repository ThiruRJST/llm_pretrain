import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#Block:
    # Masked Multi-head Attention
    # Residual connection
    # Layer Normalization
    # Feed Forward
    # Residual connection
    # Layer Normalization

#Output Head:
    # Linear layer
    # Softmax
class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads: int, dim: int, dropout:float):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.size()
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        #attention scores
        attention = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
            attention.masked_fill_(mask[None, None, ...], float("-inf"))
        
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_dropout(attention)

        out = (attention @ v).transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.proj(out)
        out = self.out_dropout(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, dim: int = 512) -> None:
        super().__init__()

        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Transformer(nn.Module):
    def __init__(self, dim:int=512, num_heads:int=8, dropout:float=0.1):
        super().__init__()

        self.attn_norm = nn.LayerNorm(dim)
        self.attention = MultiHeadAttention(dim=dim, num_heads=num_heads, dropout=dropout)

        self.ff = FeedForward(dim)
        self.ff_norm = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):

        residual = x
        x = self.attn_norm(x)
        x = self.attention(x, mask=mask)
        x = residual + self.dropout(x)
        
        residual = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = residual + self.dropout(x)
        return x
    
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, context_length: int = 256):
        super().__init__()
        
        # Create positional encodings
        pe = torch.zeros(context_length, dim)
        position = torch.arange(0, context_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, dim]
        return self.pe[:, :x.size(1)]


class GPT(nn.Module):
    def __init__(self, num_tokens: int, dim: int, num_layers:int = 6, context_length:int = 256):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = SinusoidalPositionalEncoding(dim=dim, context_length=context_length)
        self.transformer = nn.ModuleList([Transformer(dim=dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        wte = self.token_emb(x)
        wpe = self.pos_emb(x)
        x = wte + wpe

        x = self.dropout(x)
        for layer in self.transformer:
            x = layer(x)
        x = self.final_norm(x)
        x = self.head(x)
        return x