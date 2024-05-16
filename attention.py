import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import math
import copy

def attention(q, k, v, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k) #batch_size x seq_length x seq_length
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    print("Finish calculate attention score")
    return output, scores

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0 # "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.d_k = d_model//heads
        self.h = heads
        self.attn = None

        # tạo ra 3 ma trận trọng số là q_linear, k_linear, v_linear
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        """
        q: batch_size x seq_length x d_model
        k: batch_size x seq_length x d_model
        v: batch_size x seq_length x d_model
        output: batch_size x seq_length x d_model
        """
        bs = q.size(0)
        # size: batch_size x 1 x head x d_k
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # size: batch_size x head x 1 x d_k
        # tính attention score
        scores, self.attn = attention(q, k, v, self.dropout)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        print("Finish Multihead attention")
        return output
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        # self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        # x2 = self.norm_1(x) # neu da normalize tu truoc thi ko can norm nua
        x2 = x
        x = x + self.dropout_1(self.attn(x2,x2,x2))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

# checked, well run
class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Linear(input_dim, d_model) # Change dimension of raw data to d_model
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        x = self.embed(src)
        for i in range(self.N):
            x = self.layers[i](x)
        
        print("Finish encoder")
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs):
        x2 = self.norm_1(x)
        x = x + self.dropout(self.attn_1(x2, x2, x2))
        x2 = self.norm_2(x)

        x = x + self.dropout(self.attn_2(x2, e_outputs, e_outputs))
        x2 = self.norm_3(x)
        x = x + self.dropout(self.ff(x2))
        return x

class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Linear(output_dim, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, trg, e_outputs):
        x = self.embed(trg)
        for i in range(self.N):
            # x = self.layers[i](x, e_outputs, src_mask, trg_mask)
            x = self.layers[i](x, e_outputs)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(input_dim, d_model, N, heads, dropout)
        self.decoder = Decoder(output_dim, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, output_dim)
    def forward(self, src, trg):
        e_outputs = self.encoder(src)
        d_output = self.decoder(trg, e_outputs)
        output = self.out(d_output)
        return output