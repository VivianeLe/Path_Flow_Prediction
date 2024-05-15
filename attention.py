import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import math
import copy


def attention(q, k, v, mask=None, dropout=None):
    """
    q: batch_size x head x seq_length x d_model
    k: batch_size x head x seq_length x d_model
    v: batch_size x head x seq_length x d_model
    mask: batch_size x 1 x 1 x seq_length
    output: batch_size x head x seq_length x d_model
    score: batch_size x head x seq_length x seq_length
    """

    # attention score được tính bằng cách nhân q với k
    scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(q.size(-1))

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask==0, -1e9)
    # xong rồi thì chuẩn hóa bằng softmax
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
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

    def forward(self, q, k, v, mask=None):
        """
        q: batch_size x seq_length x d_model
        k: batch_size x seq_length x d_model
        v: batch_size x seq_length x d_model
        mask: batch_size x 1 x seq_length
        output: batch_size x seq_length x d_model
        """
        bs = q.size(0)
        # nhân ma trận trọng số q_linear, k_linear, v_linear với dữ liệu đầu vào q, k, v
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # tính attention score
        scores, self.attn = attention(q, k, v, mask, self.dropout)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)
        return output


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        # Standardize 
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    

class FeedForward(nn.Module):
    """ Trong kiến trúc của chúng ta có tầng linear
    """
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

def generate_mask(src, tgt):
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask
    return src_mask, tgt_mask

class Embedding(nn.Module):
    def __init__(self, data_dim, d_model):
        super().__init__()
        self.data_dim = data_dim
        self.d_model = d_model

        self.ln = nn.Linear(data_dim, d_model)
    
    def forward(self, x):
        x1 = x.unsqueeze(0) # convert to 3D tensor
        return self.ln(x1)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        # self.norm_1 = Norm(d_model)
        # self.norm_2 = Norm(d_model)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        x: batch_size x seq_length x d_model
        mask: batch_size x 1 x seq_length
        output: batch_size x seq_length x d_model
        """

        x2 = self.norm_1(x)
        # residual connection
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        # residual connection
        x = x + self.dropout_2(self.ff(x2))
        return x

# This function create N clones of an origin module
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# checked, well run
class Encoder(nn.Module):
    """ 1 encoder has many encoder layers
    """
    def __init__(self, input_dim, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Linear(input_dim, d_model) # Change dimension of raw data to d_model
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        # self.norm = Norm(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        """
        src: batch_size x seq_length
        mask: batch_size x 1 x seq_length
        output: batch_size x seq_length x d_model
        """
        x = self.embed(src)
        # Feed data to several encoder layers
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()

        # self.norm_1 = Norm(d_model)
        # self.norm_2 = Norm(d_model)
        # self.norm_3 = Norm(d_model)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        """
        x: batch_size x seq_length x d_model (target size: 1 x 25 x d_model)
        e_outputs: batch_size x seq_length x d_model (encoder output)
        src_mask: batch_size x 1 x seq_length
        trg_mask: batch_size x 1 x seq_length
        """
        x2 = self.norm_1(x)
        # multihead attention thứ nhất, chú ý các gia tri ở target
        x = x + self.dropout(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        # masked multihead attention thứ 2. k, v là giá trị output của mô hình encoder
        x = x + self.dropout(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout(self.ff(x2))
        return x


class Decoder(nn.Module):
    """ 1 decoder has many decoder layers
    N: number of decoder layer
    """
    def __init__(self, output_dim, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Linear(output_dim, d_model) # Change dimension of raw data to d_model (change from 3 to 512)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        # self.norm = Norm(d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        """
        trg: batch_size x seq_length
        e_outputs: batch_size x seq_length x d_model
        src_mask: batch_size x 1 x seq_length
        trg_mask: batch_size x 1 x seq_length
        output: batch_size x seq_length x d_model
        """
        x = self.embed(trg)
        # x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(input_dim, d_model, N, heads, dropout)
        self.decoder = Decoder(output_dim, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, output_dim)
    def forward(self, src, trg, src_mask, trg_mask):
        """
        src: batch_size x seq_length
        trg: batch_size x seq_length
        src_mask: batch_size x 1 x seq_length
        trg_mask batch_size x 1 x seq_length
        output: batch_size x seq_length x output_dim (1 x 625x3)
        """
        e_outputs = self.encoder(src, src_mask)

        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

# class EncoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, dropout):
#         super(EncoderLayer, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads)
#         # self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
#         self.feed_forward = FeedForward(d_model, dropout=dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x, mask):
#         attn_output = self.self_attn(x, x, x, mask)
#         x = self.norm1(x + self.dropout(attn_output))
#         ff_output = self.feed_forward(x)
#         x = self.norm2(x + self.dropout(ff_output))
#         return x

# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, dropout):
#         super(DecoderLayer, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads)
#         self.cross_attn = MultiHeadAttention(d_model, num_heads)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.ff = FeedForward(d_model, dropout=dropout)
        
#     def forward(self, x, enc_output, src_mask, tgt_mask):
#         attn_output = self.self_attn(x, x, x, tgt_mask)
#         x = self.norm1(x + self.dropout(attn_output))
#         attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
#         x = self.norm2(x + self.dropout(attn_output))
#         ff_output = self.feed_forward(x)
#         x = self.norm3(x + self.dropout(ff_output))
#         return x
    

# class Transformer(nn.Module):
#     def __init__(self, input_dim, output_dim, d_model, num_heads, num_layers, dropout):
#         super(Transformer, self).__init__()
#         self.encoder_embedding = nn.Linear(input_dim, d_model)
#         self.decoder_embedding = nn.Linear(output_dim, d_model)
#         # self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
#         # self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
#         # self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

#         self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
#         self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])

#         self.fc = nn.Linear(d_model, output_dim)
#         self.dropout = nn.Dropout(dropout)

    # def generate_mask(self, src, tgt):
    #     src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    #     tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
    #     seq_length = tgt.size(1)
    #     nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    #     tgt_mask = tgt_mask & nopeak_mask
    #     return src_mask, tgt_mask

#     def forward(self, src, tgt):
#         src_mask, tgt_mask = self.generate_mask(src, tgt)
#         src_embedded = self.dropout(self.encoder_embedding(src))
#         tgt_embedded = self.dropout(self.decoder_embedding(tgt))

#         enc_output = src_embedded
#         for enc_layer in self.encoder_layers:
#             enc_output = enc_layer(enc_output, src_mask)

#         dec_output = tgt_embedded
#         for dec_layer in self.decoder_layers:
#             dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

#         output = self.fc(dec_output)
#         return output