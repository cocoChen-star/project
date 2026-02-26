import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, mlp_hidden, sign_num_layers):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(in_channels, mlp_hidden) for _ in range(sign_num_layers + 1)])
        self.lin = nn.Linear((sign_num_layers + 1) * mlp_hidden, hidden_channels)
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, xs):
        outs = [F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training) for x, lin in zip(xs, self.lins)]
        x = torch.cat(outs, dim=-1)
        return torch.log_softmax(self.lin(x), dim=-1)

class TransConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads) if use_weight else None
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight and self.Wv:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, output_attn=False):
        qs = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        ks = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            vs = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            vs = source_input.reshape(-1, 1, self.out_channels)
        qs = qs / torch.norm(qs, p=2)
        ks = ks / torch.norm(ks, p=2)
        N = qs.shape[0]
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)
        attention_num += N * vs
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)
        attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer
        final_output = attn_output.mean(dim=1)
        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)
            attention = attention / normalizer
            return final_output, attention
        return final_output

class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=True):
        super().__init__()
        self.convs = nn.ModuleList([TransConvLayer(hidden_channels, hidden_channels, num_heads, use_weight) for _ in range(num_layers)])
        self.fcs = nn.ModuleList([nn.Linear(in_channels, hidden_channels)])
        self.bns = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers + 1)])
        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv, bn, fc in zip(self.convs, self.bns, self.fcs):
            conv.reset_parameters()
            bn.reset_parameters()
            fc.reset_parameters()

    def forward(self, x):
        layer_ = []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, x)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)
        return x

class Proposed(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, trans_num_layers, trans_num_heads, trans_dropout,
                 trans_use_bn=True, trans_use_residual=True, trans_use_weight=True, trans_use_act=True,
                 gnn_num_layers=3, gnn_dropout=0.5, use_graph=True, graph_weight=0.5, aggregate='add', mlp_hidden=32, sign_num_layers=3):
        super().__init__()
        self.trans_conv = TransConv(in_channels, hidden_channels, trans_num_layers, trans_num_heads, trans_dropout,
                                    trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act)
        self.graph_conv = MLP(in_channels, hidden_channels, gnn_num_layers, gnn_dropout, mlp_hidden, sign_num_layers)
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.aggregate = aggregate
        self.fc = nn.Linear(hidden_channels if aggregate == 'add' else 2 * hidden_channels, out_channels)
        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.graph_conv.parameters()) + list(self.fc.parameters())

    def forward(self, x, edge_index, xs):
        x1 = self.trans_conv(x)
        x = x1 if not self.use_graph else (
            self.graph_weight * self.graph_conv(xs) + (1 - self.graph_weight) * x1 if self.aggregate == 'add'
            else torch.cat((x1, self.graph_conv(xs)), dim=1))
        return self.fc(x)

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()