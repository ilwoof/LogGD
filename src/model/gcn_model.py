# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SGConv
from torch_geometric.nn.conv.gcn2_conv import GCN2Conv
from torch_geometric.nn.conv.gen_conv import GENConv
from torch_geometric.nn.glob.glob import global_add_pool
# from torch_geometric.nn.aggr import Aggregation, AttentionalAggregation
# from torch_geometric.transforms import AddRandomWalkPE
# from src.configuration import device


class EncoderNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
            self, num_node_type, num_in_degree, num_out_degree, hidden_dim
    ):
        super(EncoderNodeFeature, self).__init__()

        if num_node_type < 0:
            num_node_type = -num_node_type
            self.node_encoder = nn.Linear(num_node_type, hidden_dim)
        else:
            self.node_encoder = nn.Embedding(num_node_type, hidden_dim)

        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim)
        # self.positional_encoder = nn.Linear(10, hidden_dim)

        torch.nn.init.xavier_normal_(self.node_encoder.weight.data)
        torch.nn.init.xavier_normal_(self.in_degree_encoder.weight.data)
        torch.nn.init.xavier_normal_(self.out_degree_encoder.weight.data)
        # torch.nn.init.xavier_normal_(self.positional_encoder.weight.data)

    def forward(self, batched_data):
        x, in_degree, out_degree = (
            batched_data["x"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )

        # position = dgl.laplacian_pe(batched_g, 10).to(device)

        # node feature
        if isinstance(self.node_encoder, nn.Linear):
            node_feature = self.node_encoder(x)  # [node_dim, n_hidden]
        else:
            node_feature = self.node_encoder(x).sum(dim=-2)  # [n_node, n_hidden]

        node_feature = (
                node_feature
                + self.in_degree_encoder(in_degree)
                + self.out_degree_encoder(out_degree)
                # + self.positional_encoder(position)
        )

        return node_feature


class GCNNetwork(nn.Module):
    def __init__(
            self,
            out_dim=1,
            d_model=512,
            num_layer=6,
            dropout=0.1,
            num_node_type=25,
            perturb_noise=0.0,
            num_last_mlp=0,
            gnn_type='dgcn',
    ):
        super().__init__()
        self.perturb_noise = perturb_noise
        self.task_token = nn.Embedding(1, d_model, padding_idx=-1)

        if num_node_type < 0:
            num_node_type = -num_node_type
            self.node_emb = nn.Linear(num_node_type, d_model)
        else:
            self.node_emb = nn.Embedding(num_node_type, d_model)

        self.encode_node = EncoderNodeFeature(num_node_type, num_in_degree=256, num_out_degree=256, hidden_dim=d_model)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=d_model,
                    ffn_size=d_model,
                    gnn_type=gnn_type,
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
        self.readout_layer = MLPReadout(num_layer*d_model, out_dim)

    def encode_node(self, x):
        if isinstance(self.node_emb, nn.Linear):
            return self.node_emb(x)
        else:
            return self.node_emb.weight[x].sum(dim=2)

    def forward(self, batched_data):

        x = self.encode_node(batched_data.x)

        if self.training:
            perturb = torch.empty_like(x).uniform_(
                -self.perturb_noise, self.perturb_noise
            )
            x = x + perturb

        edge_index = batched_data.edge_index

        edge_attr = batched_data.edge_attr.unsqueeze(-1)

        out_all = []

        for i, enc_layer in enumerate(self.layers):
            x = enc_layer(x, edge_index, edge_attr.float())
            output = global_add_pool(x, batched_data.batch)
            out_all.append(output)

        output = torch.cat(out_all, dim=1)

        # output = self.final_ln(x)
        # output = self.last_mlp(output)
        # output = self.linear(output)

        output = self.readout_layer(output)

        return output

class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1, n_layers=3):
        super().__init__()
        fc_layers = [nn.Linear(input_dim // 2 ** n, input_dim // 2 ** (n + 1), bias=True) for n in range(n_layers)]
        fc_layers.append(nn.Linear(input_dim // 2 ** n_layers, output_dim, bias=True))
        self.fc_layers = nn.ModuleList(fc_layers)
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = nn.GELU()

    def forward(self, x):
        y = x
        for n in range(self.n_layers):
            y = F.dropout(y, self.dropout, training=self.training)
            y = self.fc_layers[n](y)
            y = self.activation(y) 
        y = self.fc_layers[self.n_layers](y)
        return y

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
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
    def __init__(self, hidden_size, ffn_size, gnn_type='deep_gcn'):
        super(EncoderLayer, self).__init__()
        self.gnn_type = gnn_type
        self.self_conv_norm = nn.LayerNorm(hidden_size)
        if gnn_type == 'gcn2':
            # GCNII
            self.self_conv = GCN2Conv(channels=hidden_size, alpha=0.1)
        elif gnn_type == 'dgcn':
            # DeepGCN
            self.self_conv = GENConv(in_channels=hidden_size, out_channels=hidden_size, num_layers=1)
        else:
            raise ValueError(f"The specified model is not supported")

        self.weight_linear = nn.Linear(1, hidden_size)

    def forward(self, x, edge_index, weight):
        y = self.self_conv_norm(x)
        if self.gnn_type == 'gcn2':
            y = self.self_conv(y, y, edge_index, weight)
        else:
            y = self.self_conv(y, edge_index, self.weight_linear(weight))
        y = F.relu(y)

        return y
