# -*- coding: utf-8 -*-
"""
Created on Tue April 07 17:16:36 2022

@author: Yongzheng Xie
"""

import dgl
import math
import torch
import torch.nn as nn
from torch.nn import init
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers, num_last_mlp=3,
                 add_self=False, ln=True, dropout=0.1, args=None):
        super(GcnEncoderGraph, self).__init__()

        self.add_self = add_self
        self.ln = ln
        self.num_layers = num_layers
        self.num_last_mlp = num_last_mlp
        self.dropout = dropout
        self.bias = True

        if args is not None:
            self.bias = args.bias

        self.node_emb = nn.Linear(input_dim, hidden_dim)

        self.act = nn.ReLU()

        self.layers = nn.ModuleList(
            [
                # norm:  'none', 'right', 'left', 'both'
                [GraphConv(in_feats=hidden_dim, out_feats=hidden_dim, norm='both', weight=True,
                           bias=self.bias, activation=self.act, allow_zero_in_degree=True)
                 for _ in range(num_layers)]
            ]
        )

        self.layer_norms = nn.ModuleList()

        for i in range(self.num_layers):
            self.layer_norms.append(nn.LayerNorm(hidden_dim, elementwise_affine=True))

        self.mlp_norm = nn.LayerNorm(hidden_dim)

        self.last_mlp = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_last_mlp)
            ]
        )
        self.classify = nn.Linear(hidden_dim, out_dim)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, g):

        x = self.encode_node(g.ndata['feat'])
        edge_weight = g.edata['weight']

        for i in range(self.num_layers):
            x = self.layer_norms[i](x) if self.ln else x
            if self.add_self:
                x = x + self.conv_block[i](g, x, edge_weight=edge_weight)
            else:
                x = self.conv_block[i](g, x, edge_weight=edge_weight)
            if self.dropout > 0.001:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.act(x)
        if self.num_last_mlp > 0:
            x = self.mlp_norm(x) if self.ln else x
            x = self.last_mlp(x)
            if self.dropout > 0.001:
                x = F.dropout(x, p=self.dropout, training=self.training)
        g.ndata['feat'] = x
        out = dgl.mean_nodes(g, 'feat')
        return self.classify(out)
