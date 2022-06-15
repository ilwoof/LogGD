# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class GCNNetwork(nn.Module):
    def __init__(
        self,
        out_dim=1,
        num_layer=6,
        d_model=512,
        dropout=0.0,
        num_node_type=25,
        perturb_noise=0.0,
        num_last_mlp=3,
        concat=False,
        ln=True,
        args=None
    ):
        super().__init__()
        self.perturb_noise = perturb_noise
        add_self = not concat
        self.bias = True
        if args is not None:
            self.bias = args.bias

        if num_node_type < 0:
            num_node_type = -num_node_type
            self.node_emb = nn.Linear(num_node_type, d_model)
        else:
            self.node_emb = nn.Embedding(num_node_type, d_model)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    input_size=d_model,
                    output_size=d_model,
                    add_self=add_self,
                    normalize=ln,
                    dropout_ratio=dropout,
                    bias=self.bias
                )
                for _ in range(num_layer)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.last_mlp = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
                for _ in range(num_last_mlp)
            ]
        )
        self.linear = nn.Linear(d_model, out_dim)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def encode_node(self, data):
        if isinstance(self.node_emb, nn.Linear):
            return self.node_emb(data.x)
        else:
            return self.node_emb.weight[data.x].sum(dim=2)

    def forward(self, data):
        x = self.encode_node(data)
        adj = data.adj

        if self.training:
            perturb = torch.empty_like(x).uniform_(
                -self.perturb_noise, self.perturb_noise
            )
            x = x + perturb

        for i, enc_layer in enumerate(self.layers):
            x = enc_layer(x, adj)

        output = self.final_ln(x)
        output = self.last_mlp(output)
        output = self.linear(output)
        return torch.mean(output, dim=1)


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout_ratio=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout_ratio
        if dropout_ratio > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout_ratio)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, input_size, output_size, add_self, normalize, dropout_ratio, bias):
        super(EncoderLayer, self).__init__()

        self.self_cov_layer = GraphConv(input_dim=input_size, output_dim=output_size, add_self=add_self,
                                        normalize_embedding=normalize, dropout_ratio=dropout_ratio, bias=bias)

    def forward(self, x, adj):
        x = self.self_cov_layer(x, adj.float())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x
