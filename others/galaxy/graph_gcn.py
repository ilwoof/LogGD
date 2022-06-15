# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.nn import GENConv

class GENGCNNetwork(nn.Module):
    def __init__(
        self,
        out_dim=1,
        num_layer=6,
        d_model=512,
        num_node_type=25,
        perturb_noise=0.0,
        num_last_mlp=3,
    ):
        super().__init__()
        self.perturb_noise = perturb_noise

        if num_node_type < 0:
            num_node_type = -num_node_type
            self.node_emb = nn.Linear(num_node_type, d_model)
        else:
            self.node_emb = nn.Embedding(num_node_type, d_model)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    input_size=d_model,
                    output_size=d_model
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
            if isinstance(m, GENGCNNetwork):
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

        if self.training:
            perturb = torch.empty_like(x).uniform_(
                -self.perturb_noise, self.perturb_noise
            )
            x = x + perturb

        for i, enc_layer in enumerate(self.layers):
            x = enc_layer(x)

        output = self.final_ln(x)
        output = self.last_mlp(output)
        output = self.linear(output)
        return torch.mean(output, dim=1)


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
    def __init__(self, input_size, output_size):
        super(EncoderLayer, self).__init__()

        self.self_cov_layer = GENConv(input_size, output_size, aggr='softmax')

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.self_cov_layer(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x
