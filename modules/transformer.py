import torch
from torch import nn
import torch.nn.functional as F
from modules.multihead_attention import MultiheadAttention
import math

import parameters
import random
import numpy as np

# Code adapted from the yaohungt repo.


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        """
        Transformer encoder consisting of N layers. Each layer is a TransformerEncoderLayer.
        @param embed_dim: input embedding
        @param num_heads: number of heads
        @param layers: number of layers
        @param attn_dropout: dropout applied on the attention weights
        @param relu_dropout: dropout applied on the first layer of the residual block
        @param res_dropout: dropout applied on the residual block
        @param embed_dropout: dropout applied on the residual block
        @param attn_mask: whether to apply mask on the attention weights
        """
        super().__init__()
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k=None, x_in_v=None, return_=False):
        """
        @param x_in: embedded input of shape (src_len, batch, embed_dim)
        @param x_in_k: embedded input of shape (src_len, batch, embed_dim)
        @param x_in_v: embedded input of shape (src_len, batch, embed_dim)
        @param return_: whether to return the weight list
        @return: the last encoder layer's output of shape (src_len, batch, embed_dim).
            if return_=True, return tuple (output, weights)
        """
        # embed tokens
        x = self.embed_scale * x_in
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        
        # encoder layers
        intermediates = [x]
        matrixs = []
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x, matrix = layer(x, x_k, x_v, return_=True)
            else:
                x = layer(x)
            intermediates.append(x)
            if return_:
                matrixs.append(matrix)

        if self.normalize:
            x = self.layer_norm(x)

        if return_:
            return x, matrixs
        else:
            return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        """
        Encoder layer block
        @param embed_dim: input embedding
        @param num_heads: number of heads
        @param attn_dropout: dropout applied on the attention weights
        @param relu_dropout: dropout applied on the first layer of the residual block
        @param res_dropout: dropout applied on the residual block
        @param attn_mask: whether to apply mask on the attention weights
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None, return_=False):
        """
        @param x: (seq_len, batch, embed_dim)
        @param x_k: (seq_len, batch, embed_dim)
        @param x_v: (seq_len, batch, embed_dim)
        @param return_: whether to return the weight list
        @return: encoded output of shape (batch, src_len, embed_dim).
            if return_=True, return tuple (output, weight)
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        # Cross-modal attention
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True) 
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        # Position-wise feed forward
        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        if return_:
            return x, _
        else:
            return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.to(parameters.device)
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m
