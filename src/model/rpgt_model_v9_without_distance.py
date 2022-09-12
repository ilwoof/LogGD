# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys

sys.path.append("../")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.configuration import device


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
            self.node_encoder = nn.Linear(num_node_type, hidden_dim, bias=False)
        else:
            self.node_encoder = nn.Embedding(num_node_type, hidden_dim)

        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)

        torch.nn.init.xavier_normal_(self.node_encoder.weight.data)

    def forward(self, batched_data):
        x, in_degree, out_degree = (
            batched_data["x"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )

        # node feature
        if isinstance(self.node_encoder, nn.Linear):
            node_feature = self.node_encoder(x)
        else:
            node_feature = self.node_encoder(x).sum(dim=-2)

        node_feature = (
                node_feature
                + self.in_degree_encoder(in_degree)
                + self.out_degree_encoder(out_degree)
        )

        return node_feature


class GRPENetwork(nn.Module):
    def __init__(
            self,
            out_dim=1,
            num_layer=6,
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            attention_dropout=0.1,
            max_hop=256,
            num_node_type=25,
            num_edge_type=25,
            perturb_noise=0.0,
            num_last_mlp=0,
    ):
        super().__init__()
        self.perturb_noise = perturb_noise
        self.max_hop = max_hop
        self.num_edge_type = num_edge_type

        self.encode_node = EncoderNodeFeature(num_node_type, num_in_degree=256, num_out_degree=256, hidden_dim=d_model)

        # if num_node_type < 0:
        #     num_node_dim = -num_node_type
        #     self.node_emb = nn.Linear(num_node_dim, d_model)
        # else:
        #     self.node_emb = nn.Embedding(num_node_type, d_model)

        self.UNREACHABLE_DISTANCE = max_hop + 1

        # query_hop_emb: Query Structure Embedding
        # key_hop_emb: Key Structure Embedding
        # value_hop_emb: Value Structure Embedding

        self.query_hop_emb = nn.Embedding(max_hop + 2, d_model)
        self.key_hop_emb = nn.Embedding(max_hop + 2, d_model)
        self.value_hop_emb = nn.Embedding(max_hop + 2, d_model)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=d_model,
                    ffn_size=dim_feedforward,
                    dropout_rate=dropout,
                    attention_dropout_rate=attention_dropout,
                    num_heads=nhead,
                )
                for _ in range(num_layer)
            ]
        )
        self.final_ln = nn.LayerNorm(2 * d_model)
        # self.final_ln = nn.LayerNorm(d_model)
        self.last_mlp = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
                for _ in range(num_last_mlp)
            ]
        )
        self.linear = nn.Linear(2 * d_model, d_model)
        self.readout_layer = MLPReadout(2 * d_model, out_dim)

    # def encode_node(self, data):
    #     if isinstance(self.node_emb, nn.Linear):
    #         return self.node_emb(data.x)
    #     else:
    #         return self.node_emb.weight[data.x].sum(dim=2)

    def forward(self, data):
        mask = data.mask
        x = self.encode_node(data)

        # calculate occurrence of nodes in the sequence graph
        nodes_occurrences = data.edge_weight.sum(dim=-2)

        if self.training:
            perturb = torch.empty_like(x).uniform_(
                -self.perturb_noise, self.perturb_noise
            )
            x = x + perturb

        distance = data.distance.clamp(
            max=self.max_hop
        )  # max_hop is $\mathcal{P}_\text{far}$
        distance[distance == -1] = self.UNREACHABLE_DISTANCE

        for i, enc_layer in enumerate(self.layers):
            x = enc_layer(
                x,
                self.query_hop_emb.weight,
                # self.key_hop_emb.weight,
                # self.value_hop_emb.weight,
                self.query_hop_emb.weight,
                self.query_hop_emb.weight,
                distance,
                nodes_occurrences,
                mask=mask,
            )
        # output = self.last_mlp(output)
        # output = self.linear(output)
        # return output

        output = torch.cat([torch.sum(x, dim=1), torch.max(x, dim=1).values], dim=1)
        output = self.final_ln(output)

        # output = self.final_ln(torch.sum(x, dim=1))
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


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(
            self,
            q,
            k,
            v,
            query_hop_emb,
            key_hop_emb,
            value_hop_emb,
            distance,
            mask=None,
    ):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        k = k.transpose(1, 2)  # [b, h, k_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]

        # sequence_length = v.shape[2]
        # num_hop_types = query_hop_emb.shape[0]
        #
        # query_hop_emb = query_hop_emb.view(
        #     1, num_hop_types, self.num_heads, self.att_size
        # ).transpose(1, 2)
        #
        # key_hop_emb = key_hop_emb.view(
        #     1, num_hop_types, self.num_heads, self.att_size
        # ).transpose(1, 2)
        #
        # query_hop = torch.matmul(q, query_hop_emb.transpose(2, 3))
        # query_hop = torch.gather(
        #     query_hop, 3, distance.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        # )
        #
        # key_hop = torch.matmul(k, key_hop_emb.transpose(2, 3))
        # # key_hop = torch.gather(
        # #     key_hop, 3, distance.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        # # )
        # key_hop = torch.gather(
        #     key_hop.transpose(2, 3), 2, distance.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        # )
        #
        # spatial_bias = query_hop + key_hop

        x = torch.matmul(q, k.transpose(2, 3))  # + spatial_bias
        x = x * self.scale
        if mask is not None:
            x = x.masked_fill(mask[:, 0, :].view(mask.shape[0], 1, 1, mask.shape[2]), float("-inf"))

        x = torch.softmax(x, dim=3)
        if mask is not None:
            cleaning_value = torch.ones_like(x, dtype=x.dtype, device=x.device)
            cleaning_value = cleaning_value.masked_fill(mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1), 0.0)
            x = x * cleaning_value

        x = self.att_dropout(x)

        # spatial_bias = spatial_bias * self.scale
        # if mask is not None:
        #     spatial_bias = spatial_bias.masked_fill(mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1), 0.0)
        #
        # spatial_bias = self.att_dropout(spatial_bias)

        # value_hop_emb = value_hop_emb.view(
        #     1, num_hop_types, self.num_heads, self.att_size
        # ).transpose(1, 2)
        #
        # value_hop_att = torch.zeros(
        #     (batch_size, self.num_heads, sequence_length, num_hop_types),
        #     device=value_hop_emb.device,
        # )
        # value_hop_att = torch.scatter_add(
        #     value_hop_att, 3, distance.unsqueeze(1).repeat(1, self.num_heads, 1, 1), x   # spatial_bias
        # )
        x = (
            torch.matmul(x, v)
            # + torch.matmul(value_hop_att, value_hop_emb)
        )
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(
            self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads
    ):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size,
            attention_dropout_rate,
            num_heads,
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(
            self,
            x,
            query_hop_emb,
            key_hop_emb,
            value_hop_emb,
            distance,
            nodes_occurrences,
            mask=None,
    ):
        k = v = self.self_attention_norm(nodes_occurrences.view(-1, x.shape[1], 1) * x)
        q = self.self_attention_norm(x)

        y = self.self_attention(
            q,
            k,
            v,
            query_hop_emb,
            key_hop_emb,
            value_hop_emb,
            distance,
            mask=mask,
        )
        y = self.self_attention_dropout(y)
        x = k + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
