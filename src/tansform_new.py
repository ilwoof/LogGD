import torch
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.loader.dataloader import Collater as GraphCollater


class ShortestPathGenerator:
    def __init__(self, directed=True):
        self.directed = directed

    def __call__(self, data):
        row = data.edge_index[0].numpy()
        col = data.edge_index[1].numpy()
        weight = np.ones_like(row)
        # weight = data.edge_weight.numpy()

        graph = csr_matrix((weight, (row, col)), shape=(len(data.x), len(data.x)))
        dist_matrix, _ = shortest_path(
            csgraph=graph, directed=self.directed, return_predecessors=True
        )
        data["distance"] = torch.from_numpy(dist_matrix)
        return data

class OneHotEdgeAttr:
    def __init__(self, max_range=4) -> None:
        self.max_range = max_range

    def __call__(self, data):
        x = data["edge_attr"]
        if len(x.shape) == 1:
            return data

        offset = torch.ones((1, x.shape[1]), dtype=torch.float32)
        offset[:, 1:] = self.max_range
        offset = torch.cumprod(offset, dim=1)
        x = (x * offset).sum(dim=1)
        data["edge_attr"] = x
        return data


class LogCollator(object):
    def __init__(self, max_node=None) -> None:
        super().__init__()
        self.collator = GraphCollater(
            [],
            exclude_keys=[
                # "x,",
                "distance",
                # "edge_index",
                # "edge_attr",
                "in_degree",
                "out_degree",
                "edge_weight",
            ],
        )
        self.max_node = max_node

    def __call__(self, batch):
        if self.max_node is not None:
            batch = [b for b in batch if b["x"].shape[0] <= self.max_node]

        node = [b["x"] for b in batch]
        distance = [b["distance"] for b in batch]

        if len(distance[0].shape) == 1:
            distance = [d.view(n.shape[0], n.shape[0]) for d, n in zip(distance, node)]

        edge_index = [b["edge_index"] for b in batch]
        edge_attr = [b["edge_attr"] for b in batch]
        edge_weight = [b["edge_weight"] for b in batch]
        max_num_node = max(d.shape[0] for d in distance)

        gathered_node = []
        gathered_adj = []
        gathered_distance = []
        # gathered_edge_attr = []
        gathered_edge_weight = []
        gathered_in_degree = []
        gathered_out_degree = []
        mask = []

        # for n, d, ei, ea, ew in zip(node, distance, edge_index, edge_attr, edge_weight):
        for n, d, ei, ew in zip(node, distance, edge_index, edge_weight):
            # m = torch.zeros(max_num_node, dtype=torch.bool)
            # m[n.shape[0] :] = 1

            m = torch.zeros(max_num_node, max_num_node, dtype=torch.bool)
            m[n.shape[0]:, :] = 1
            m[:, n.shape[0]:] = 1

            new_n = torch.zeros((max_num_node, n.shape[1]), dtype=torch.float32)
            new_n[: n.shape[0]] = n

            new_d = -torch.ones((max_num_node, max_num_node), dtype=torch.long)

            new_d[: d.shape[0], : d.shape[1]] = d
            new_d[new_d < 0] = -1

            new_adj = torch.zeros((max_num_node, max_num_node), dtype=torch.long)
            new_adj[ei[0], ei[1]] = 1

            # new_ea = -torch.ones((max_num_node, max_num_node),  dtype=torch.long)
            # new_ea[ei[0], ei[1]] = ea

            new_ew = torch.zeros((max_num_node, max_num_node), dtype=torch.float32)
            new_ew[ei[0], ei[1]] = ew

            new_ind = torch.zeros(max_num_node, dtype=torch.long)
            new_ind[: n.shape[0]] = new_adj[: n.shape[0], : n.shape[0]].sum(dim=-2)

            new_outd = torch.zeros(max_num_node, dtype=torch.long)
            new_outd[: n.shape[0]] = new_adj[: n.shape[0], : n.shape[0]].sum(dim=-1)

            mask.append(m)
            gathered_node.append(new_n)
            gathered_adj.append(new_adj)
            gathered_distance.append(new_d)
            # gathered_edge_attr.append(new_ea)
            gathered_edge_weight.append(new_ew)
            gathered_in_degree.append(new_ind)
            gathered_out_degree.append(new_outd)

        mask = torch.stack(mask, dim=0)
        gathered_node = torch.stack(gathered_node, dim=0)
        gathered_adj = torch.stack(gathered_adj, dim=0)
        gathered_distance = torch.stack(gathered_distance, dim=0)
        # gathered_edge_attr = torch.stack(gathered_edge_attr, dim=0)
        gathered_edge_weight = torch.stack(gathered_edge_weight, dim=0)
        gathered_in_degree = torch.stack(gathered_in_degree, dim=0)
        gathered_out_degree = torch.stack(gathered_out_degree, dim=0)

        batch = self.collator(batch)
        # batch["x"] = gathered_node
        batch["adj"] = gathered_adj
        batch["mask"] = mask
        batch["distance"] = gathered_distance
        # batch["edge_attr"] = gathered_edge_attr
        batch["edge_weight"] = gathered_edge_weight
        batch["in_degree"] = gathered_in_degree
        batch["out_degree"] = gathered_out_degree
        return batch
