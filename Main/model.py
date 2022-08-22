# -*- coding: utf-8 -*-
# @Time    : 2022/7/21 17:18
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : model.py
# @Software: PyCharm
# @Note    :
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINConv, VGAE, global_mean_pool, global_add_pool, GCNConv
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops, negative_sampling, subgraph


class ResGCN(nn.Module):
    """GCN with BN and residual connection."""

    def __init__(self, dataset, hidden, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0,
                 edge_norm=True):
        super().__init__()

        # print("GFN:", gfn)

        assert num_feat_layers == 1, "more feat layers are not now supported"
        self.conv_residual = residual
        self.fc_residual = False  # no skip-connections for fc layers.
        self.res_branch = res_branch
        self.collapse = collapse
        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout

        # GCNConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        if "xg" in dataset[0]:  # Utilize graph level features.
            self.use_xg = True
            self.bn1_xg = BatchNorm1d(dataset[0].xg.size(1))
            self.lin1_xg = Linear(dataset[0].xg.size(1), hidden)
            self.bn2_xg = BatchNorm1d(hidden)
            self.lin2_xg = Linear(hidden, hidden)
        else:
            self.use_xg = False

        hidden_in = dataset.num_features

        self.bn_feat = BatchNorm1d(hidden_in)
        feat_gfn = True  # set true so GCNConv is feat transform
        self.conv_feat = GCNConv(hidden_in, hidden)
        if "gating" in global_pool:
            self.gating = nn.Sequential(
                Linear(hidden, hidden),
                nn.ReLU(),
                Linear(hidden, 1),
                nn.Sigmoid())
        else:
            self.gating = None
        self.bns_conv = nn.ModuleList()
        self.convs = nn.ModuleList()
        if self.res_branch == "resnet":
            for i in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GCNConv(hidden, hidden))
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GCNConv(hidden, hidden))
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GCNConv(hidden, hidden))
        else:
            for i in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GCNConv(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = nn.ModuleList()
        self.lins = nn.ModuleList()
        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        # self.lin_class = Linear(hidden, dataset.num_classes)
        self.lin_class = Linear(hidden, 1)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.0001)

        self.proj_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))

    def reset_parameters(self):
        raise NotImplemented(
            "This is prune to bugs (e.g. lead to training on test set in "
            "cross validation setting). Create a new model instance instead.")

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        if self.res_branch == "BNConvReLU":
            return self.forward_BNConvReLU(x, edge_index, batch, xg)
        else:
            raise ValueError("Unknown res_branch %s" % self.res_branch)

    def forward_BNConvReLU(self, x, edge_index, batch, xg=None):
        # embed()
        # exit()
        # print("this forward")
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        # return F.log_softmax(x, dim=-1)
        return torch.sigmoid(x).view(-1)

    def forward_last_layers(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_

        out1 = self.bn_hidden(x)
        if self.dropout > 0:
            out1 = F.dropout(out1, p=self.dropout, training=self.training)

        out2 = self.lin_class(out1)
        out3 = F.log_softmax(out2, dim=-1)
        return out1, out2, out3

    def forward_cl(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        return x

    def forward_graph_cl(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.proj_head(x)
        return x


class GIN_NodeWeightEncoder(torch.nn.Module):
    def __init__(self, dataset, dim, add_mask=False):
        super().__init__()

        num_features = dataset.num_features
        # num_features = dataset_num_features

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        # nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv3 = GINConv(nn3)
        # self.bn3 = torch.nn.BatchNorm1d(dim)

        # nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv4 = GINConv(nn4)
        # self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = None
        if add_mask == True:
            nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, 3))
            self.conv5 = GINConv(nn5)
            self.bn5 = torch.nn.BatchNorm1d(3)
        else:
            nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, 2))
            self.conv5 = GINConv(nn5)
            self.bn5 = torch.nn.BatchNorm1d(2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)

        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        # x = F.relu(self.conv3(x, edge_index))
        # x = self.bn3(x)
        # x = F.relu(self.conv4(x, edge_index))
        # x = self.bn4(x)

        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        return x


class ViewGenerator(VGAE):
    def __init__(self, dataset, dim, encoder, add_mask=False):
        self.add_mask = add_mask
        encoder = encoder(dataset, dim, self.add_mask)
        super().__init__(encoder=encoder)

    def sample_view(self, data):
        data = copy.deepcopy(data)
        edge_index = data.edge_index
        z = self.encode(data)
        # pre_recovered = self.decoder.forward_all(z)
        # exp_num = pre_recovered.sum()

        recovered = self.decoder.forward_all(z)
        exp_num = recovered.sum()
        recovered = self.decoder.forward_all(z) * (data.num_edges / float(exp_num))
        edge_selected = torch.bernoulli(recovered)
        edge_selected = edge_selected.bool()

        edge_index = edge_selected.nonzero(as_tuple=False).T
        # print(edge_selected)
        edge_index = to_undirected(edge_index)
        edge_index = add_self_loops(edge_index)[0]
        data.edge_index = edge_index
        return z, recovered, data

    def sample_partial_view(self, data):
        data = copy.deepcopy(data)
        z = self.encode(data)
        edge_index = data.edge_index

        neg_edge_index = negative_sampling(edge_index)
        joint_edge_index = torch.cat((edge_index, neg_edge_index), dim=1)
        # joint_edge_index = to_undirected(joint_edge_index)
        joint_edge_index = remove_self_loops(joint_edge_index)[0]
        # joint_edge_index = add_self_loops(joint_edge_index)[0]

        wanted_num_edges = data.num_edges // 2
        edge_weights = self.decoder.forward(z, joint_edge_index)
        exp_num_edges = edge_weights.sum()
        edge_weights *= wanted_num_edges / exp_num_edges

        edge_selected = torch.bernoulli(edge_weights)
        edge_selected = edge_selected.bool()

        edge_index = joint_edge_index[:, edge_selected]
        edge_index = to_undirected(edge_index)
        edge_index = remove_self_loops(edge_index)[0]

        data.edge_index = edge_index
        return z, None, data

    def sample_partial_view_recon(self, data, neg_edge_index):
        data = copy.deepcopy(data)
        z = self.encode(data)
        # return z, None, None

        edge_index = data.edge_index

        if neg_edge_index == None:
            neg_edge_index = negative_sampling(edge_index)

        joint_edge_index = torch.cat((edge_index, neg_edge_index), dim=1)
        # joint_edge_index = edge_index
        # joint_edge_index = to_undirected(joint_edge_index)

        # wanted_num_edges = data.num_edges // 2
        edge_weights = self.decoder.forward(z, joint_edge_index)
        edge_selected = torch.bernoulli(edge_weights)
        edge_selected = edge_selected.bool()

        edge_index = joint_edge_index[:, edge_selected]
        edge_index = to_undirected(edge_index)

        # edge_index = add_self_loops(edge_index)[0]
        # print("final edges:", edge_index.shape[1])
        data.edge_index = edge_index
        return z, neg_edge_index, data

    def sample_subgraph_view(self, data):
        data = copy.deepcopy(data)
        z = self.encode(data)
        edge_index = data.edge_index

        recovered_all = self.decoder.forward_all(z)
        recovered = self.decode(z, edge_index)
        edge_selected = torch.bernoulli(recovered)
        edge_selected = edge_selected.bool()
        edge_index = edge_index[:, edge_selected]
        edge_index = to_undirected(edge_index)

        edge_index = add_self_loops(edge_index, num_nodes=data.num_nodes)[0]

        data.edge_index = edge_index
        return z, recovered_all, data

    def forward(self, data_in, requires_grad):
        data = copy.deepcopy(data_in)

        x, edge_index = data.x, data.edge_index
        edge_attr = None
        if data.edge_attr is not None:
            edge_attr = data.edge_attr

        data.x = data.x.float()
        x = x.float()
        x.requires_grad = requires_grad

        p = self.encoder(data)
        sample = F.gumbel_softmax(p, hard=True)

        real_sample = sample[:, 0]
        attr_mask_sample = None
        if self.add_mask == True:
            attr_mask_sample = sample[:, 2]
            keep_sample = real_sample + attr_mask_sample
        else:
            keep_sample = real_sample

        keep_idx = torch.nonzero(keep_sample, as_tuple=False).view(-1, )
        edge_index, edge_attr = subgraph(keep_idx, edge_index, edge_attr, num_nodes=data.num_nodes)
        x = x * keep_sample.view(-1, 1)

        if self.add_mask == True:
            attr_mask_idx = attr_mask_sample.bool()
            token = data.x.detach().mean()
            x[attr_mask_idx] = token

        data.x = x
        data.edge_index = edge_index
        if data.edge_attr is not None:
            data.edge_attr = edge_attr

        return keep_sample, data
