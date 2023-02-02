# Courtesy of Long Range Graph Benchmarks (Dwivedi et. al, 2022) and https://github.com/vijaydwivedi75/lrgb

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import torch_geometric.graphgym.register as register
from torch_geometric.utils import remove_self_loops
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network, register_head
from torch_geometric.graphgym.models.layer import new_layer_config, MLP


class SAN(torch.nn.Module):
    """Spectral Attention Network (SAN) Graph Transformer.
    https://arxiv.org/abs/2106.03893
    """

    def __init__(self, input_channels, hidden_channels, num_layers, output_channels, heads, layerpm = 1,
                 layer_type='SANLayer', dropout=True, layer_norm = False, batch_norm = True, residual=True, head_type='default'):
        super().__init__()
        self.encoder = FeatureEncoder(input_channels)
        input_channels = self.encoder.dim_in

        if layerpm > 0:
            self.pre_mp = GNNPreMP(
                input_channels, hidden_channels, layerpm)
            input_channels = hidden_channels

        assert hidden_channels == input_channels, \
            "The inner and hidden dims must match."

        fake_edge_emb = torch.nn.Embedding(1, hidden_channels)
        # torch.nn.init.xavier_uniform_(fake_edge_emb.weight.data)
        Layer = {
            'SANLayer': SANLayer,
            'SAN2Layer': SAN2Layer,
        }.get(layer_type)
        layers = []
        for _ in range(num_layers):
            layers.append(Layer(gamma=1e-6,
                                in_dim=hidden_channels,
                                out_dim=hidden_channels,
                                num_heads=heads,
                                full_graph=True,
                                fake_edge_emb=fake_edge_emb,
                                dropout=dropout,
                                layer_norm=layer_norm,
                                batch_norm=batch_norm,
                                residual=residual))
        self.trf_layers = torch.nn.Sequential(*layers)
        register_head('default', NodeHead)
        GNNHead = register.head_dict[head_type]
        self.post_mp = GNNHead(dim_in=hidden_channels, dim_out=output_channels)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class NodeHead(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(NodeHead, self).__init__()
        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))

    def _apply_index(self, batch):
        return batch.x, batch.y

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        return pred, label



class MultiHeadAttentionLayer(nn.Module):
    """Multi-Head Graph Attention Layer.
    Ported to PyG from original repo:
    https://github.com/DevinKreuzer/SAN/blob/main/layers/graph_transformer_layer.py
    """

    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph,
                 fake_edge_emb, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gamma = gamma
        self.full_graph = full_graph

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(2, out_dim * num_heads, bias=use_bias)

        if self.full_graph:
            self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.fake_edge_emb = fake_edge_emb

        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]  # (num real edges) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]  # (num real edges) x num_heads x out_dim
        score = torch.mul(src, dest)  # element-wise multiplication

        # Scale scores by sqrt(d)
        score = score / np.sqrt(self.out_dim)

        if self.full_graph:
            fake_edge_index = negate_edge_index(batch.edge_index, batch.batch)
            src_2 = batch.K_2h[fake_edge_index[0]]  # (num fake edges) x num_heads x out_dim
            dest_2 = batch.Q_2h[fake_edge_index[1]]  # (num fake edges) x num_heads x out_dim
            score_2 = torch.mul(src_2, dest_2)

            # Scale scores by sqrt(d)
            score_2 = score_2 / np.sqrt(self.out_dim)

        # Use available edge features to modify the scores for edges
        score = torch.mul(score, batch.E)  # (num real edges) x num_heads x out_dim

        if self.full_graph:
            # E_2 is 1 x num_heads x out_dim and will be broadcast over dim=0
            score_2 = torch.mul(score_2, batch.E_2)

        if self.full_graph:
            # softmax and scaling by gamma
            score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # (num real edges) x num_heads x 1
            score_2 = torch.exp(score_2.sum(-1, keepdim=True).clamp(-5, 5))  # (num fake edges) x num_heads x 1
            score = score / (self.gamma + 1)
            score_2 = self.gamma * score_2 / (self.gamma + 1)
        else:
            score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # (num real edges) x num_heads x 1

        # Apply attention score to each source node to create edge messages
        msg = batch.V_h[batch.edge_index[0]] * score  # (num real edges) x num_heads x out_dim
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

        if self.full_graph:
            # Attention via fictional edges
            msg_2 = batch.V_h[fake_edge_index[0]] * score_2
            # Add messages along fake edges to destination nodes
            scatter(msg_2, fake_edge_index[1], dim=0, out=batch.wV, reduce='add')

        # Compute attention normalization coefficient
        batch.Z = score.new_zeros(batch.size(0), self.num_heads, 1)  # (num nodes in batch) x num_heads x 1
        scatter(score, batch.edge_index[1], dim=0, out=batch.Z, reduce='add')
        if self.full_graph:
            scatter(score_2, fake_edge_index[1], dim=0, out=batch.Z, reduce='add')

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)
        E = self.E(batch.edge_attr)

        if self.full_graph:
            Q_2h = self.Q_2(batch.x)
            K_2h = self.K_2(batch.x)
            # One embedding used for all fake edges; shape: 1 x emb_dim
            dummy_edge = self.fake_edge_emb(batch.edge_index.new_zeros(1))
            E_2 = self.E_2(dummy_edge)

        V_h = self.V(batch.x)
        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.E = E.view(-1, self.num_heads, self.out_dim)

        if self.full_graph:
            batch.Q_2h = Q_2h.view(-1, self.num_heads, self.out_dim)
            batch.K_2h = K_2h.view(-1, self.num_heads, self.out_dim)
            batch.E_2 = E_2.view(-1, self.num_heads, self.out_dim)

        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(batch)
        h_out = batch.wV / (batch.Z + 1e-6)

        return h_out


class SANLayer(nn.Module):
    """GraphTransformerLayer from SAN.
    Ported to PyG from original repo:
    https://github.com/DevinKreuzer/SAN/blob/main/layers/graph_transformer_layer.py
    """

    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph,
                 fake_edge_emb, dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.attention = MultiHeadAttentionLayer(gamma=gamma,
                                                 in_dim=in_dim,
                                                 out_dim=out_dim // num_heads,
                                                 num_heads=num_heads,
                                                 full_graph=full_graph,
                                                 fake_edge_emb=fake_edge_emb,
                                                 use_bias=use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection
        # multi-head attention out
        h_attn_out = self.attention(batch)
        # Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)
        #h = h.reshape(-1, hidden_channels)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.O_h(h)
        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_max, scatter_add


def pyg_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.
    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
            scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out


class MultiHeadAttention2Layer(nn.Module):
    """Multi-Head Graph Attention Layer.
    Ported to PyG and modified compared to the original repo:
    https://github.com/DevinKreuzer/SAN/blob/main/layers/graph_transformer_layer.py
    """

    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph,
                 fake_edge_emb, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gamma = nn.Parameter(torch.tensor(0.5, dtype=float),
                                  requires_grad=True)
        self.full_graph = full_graph

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        if self.full_graph:
            self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.fake_edge_emb = fake_edge_emb

        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]  # (num real edges) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]  # (num real edges) x num_heads x out_dim
        score = torch.mul(src, dest)  # element-wise multiplication

        # Scale scores by sqrt(d)
        score = score / np.sqrt(self.out_dim)

        if self.full_graph:
            fake_edge_index = negate_edge_index(batch.edge_index, batch.batch)
            src_2 = batch.K_2h[fake_edge_index[0]]  # (num fake edges) x num_heads x out_dim
            dest_2 = batch.Q_2h[fake_edge_index[1]]  # (num fake edges) x num_heads x out_dim
            score_2 = torch.mul(src_2, dest_2)

            # Scale scores by sqrt(d)
            score_2 = score_2 / np.sqrt(self.out_dim)

        # Use available edge features to modify the scores for edges
        score = torch.mul(score, batch.E)  # (num real edges) x num_heads x out_dim

        if self.full_graph:
            # E_2 is 1 x num_heads x out_dim and will be broadcast over dim=0
            score_2 = torch.mul(score_2, batch.E_2)

        if self.full_graph:
            # softmax and scaling by gamma
            score = pyg_softmax(score.sum(-1, keepdim=True), batch.edge_index[1])  # (num real edges) x num_heads x 1
            score_2 = pyg_softmax(score_2.sum(-1, keepdim=True), fake_edge_index[1])  # (num fake edges) x num_heads x 1
            score = score / (self.gamma + 1)
            score_2 = self.gamma * score_2 / (self.gamma + 1)
        else:
            score = pyg_softmax(score.sum(-1, keepdim=True), batch.edge_index[1])  # (num real edges) x num_heads x 1

        # Apply attention score to each source node to create edge messages
        msg = batch.V_h[batch.edge_index[0]] * score  # (num real edges) x num_heads x out_dim
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

        if self.full_graph:
            # Attention via fictional edges
            msg_2 = batch.V_h[fake_edge_index[0]] * score_2
            # Add messages along fake edges to destination nodes
            scatter(msg_2, fake_edge_index[1], dim=0, out=batch.wV, reduce='add')


    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)
        E = self.E(batch.edge_attr)

        if self.full_graph:
            Q_2h = self.Q_2(batch.x)
            K_2h = self.K_2(batch.x)
            # One embedding used for all fake edges; shape: 1 x emb_dim
            dummy_edge = self.fake_edge_emb(batch.edge_index.new_zeros(1))
            E_2 = self.E_2(dummy_edge)

        V_h = self.V(batch.x)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.E = E.view(-1, self.num_heads, self.out_dim)

        if self.full_graph:
            batch.Q_2h = Q_2h.view(-1, self.num_heads, self.out_dim)
            batch.K_2h = K_2h.view(-1, self.num_heads, self.out_dim)
            batch.E_2 = E_2.view(-1, self.num_heads, self.out_dim)

        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch)

        h_out = batch.wV

        return h_out


class SAN2Layer(nn.Module):
    """Modified GraphTransformerLayer from SAN.
    Ported to PyG from original repo:
    https://github.com/DevinKreuzer/SAN/blob/main/layers/graph_transformer_layer.py
    """

    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph,
                 fake_edge_emb, dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.attention = MultiHeadAttention2Layer(gamma=gamma,
                                                  in_dim=in_dim,
                                                  out_dim=out_dim // num_heads,
                                                  num_heads=num_heads,
                                                  full_graph=full_graph,
                                                  fake_edge_emb=fake_edge_emb,
                                                  use_bias=use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        # multi-head attention out
        h_attn_out = self.attention(batch)

        # Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual)

def negate_edge_index(edge_index, batch=None):
    """Negate batched sparse adjacency matrices given by edge indices.
    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.
    Implementation inspired by `torch_geometric.utils.to_dense_adj`
    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.
    Returns:
        Complementary edge index.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch,
                        dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short,
                         device=edge_index.device)

        # Remove existing edges from the full N x N adjacency matrix
        flattened_size = n * n
        adj = adj.view([flattened_size])
        _idx1 = idx1[idx0 == i]
        _idx2 = idx2[idx0 == i]
        idx = _idx1 * n + _idx2
        zero = torch.zeros(_idx1.numel(), dtype=torch.short,
                           device=edge_index.device)
        scatter(zero, idx, dim=0, out=adj, reduce='mul')

        # Convert to edge index format
        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_negative = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_negative
