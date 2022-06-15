# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class TransformerGCNNetwork(nn.Module):
    def __init__(
        self,
        out_dim=1,
        num_layer=6,
        d_model=512,
        nhead=8,
        dropout=0.1,
        attention_dropout=0.1,
        num_node_type=25,
        perturb_noise=0.0,
        num_last_mlp=3,
    ):
        super().__init__()
        self.perturb_noise = perturb_noise
        self.task_token = nn.Embedding(1, d_model, padding_idx=-1)

        if num_node_type < 0:
            num_node_type = -num_node_type
            self.node_emb = nn.Linear(num_node_type, d_model)
        else:
            self.node_emb = nn.Embedding(num_node_type, d_model)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=d_model,
                    ffn_size=d_model,
                    dropout_rate=dropout,
                    attention_dropout_rate=attention_dropout,
                    num_heads=nhead,
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

    def encode_node(self, data):
        if isinstance(self.node_emb, nn.Linear):
            return self.node_emb(data.x)
        else:
            return self.node_emb.weight[data.x].sum(dim=2)

    def forward(self, data):
        x = self.encode_node(data)
        edge_index = data.edge_index
        print(edge_index)
        # x = self.encode_node(data).squeeze()
        # edge_index = data.edge_index.squeeze().nonzero().t().contiguous()

        if self.training:
            perturb = torch.empty_like(x).uniform_(
                -self.perturb_noise, self.perturb_noise
            )
            x = x + perturb

        # # Append Task Token
        # x_with_task = torch.zeros(
        #     (x.shape[0], x.shape[1] + 1, x.shape[2]), dtype=x.dtype, device=x.device
        # )
        # x_with_task[:, 1:] = x
        # x_with_task[:, 0] = self.task_token.weight

        for i, enc_layer in enumerate(self.layers):
            # x_with_task = enc_layer(x_with_task)
            x = enc_layer(x, edge_index)

        # output = self.final_ln(x_with_task[:, 0])
        output = self.final_ln(x)
        output = self.last_mlp(output)
        output = self.linear(output)
        return torch.mean(output)


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
    def __init__(
        self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads
    ):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = TransformerConv(
            in_channels=hidden_size,
            out_channels=ffn_size,
            heads=num_heads,
            dropout=attention_dropout_rate,
            concat=False,
            # add_self_loops=True
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):

        y = self.self_attention_norm(x)
        y = self.self_attention(y, edge_index)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
