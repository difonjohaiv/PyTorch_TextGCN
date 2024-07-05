import math
import torch as th
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple pygGCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, infeatn, adj):
        support = th.spmm(infeatn, self.weight)
        output = th.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GCN(Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = th.relu(x)
        x = th.dropout(x, self.dropout, train=self.training)
        x = self.gc2(x, adj)
        return x

class Model(Module):

    def __init__(self,
                 input_dim: int,
                 num_hidden: int,
                 num_class: int,
                 dropout: float = 0.5,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder = GCN(nfeat=input_dim, nhid=num_hidden, nclass=num_class, dropout=dropout)
        self.tau: float = tau

        self.fc = torch.nn.Linear(num_hidden, num_class)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x = x, adj = adj)
        y_4_semi = self.fc(h)
        return h, y_4_semi

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    # NT-Xent对比学习损失
    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        # f = lambda x: torch.exp(x / self.tau)
        def f(x):
            return torch.exp(x / self.tau)

        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag() /
            (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1

        # f = lambda x: torch.exp(x / self.tau)
        def f(x):
            return torch.exp(x / self.tau)

        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag() /
                (refl_sim.sum(1) + between_sim.sum(1) -
                 refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self,
             z1: torch.Tensor,
             z2: torch.Tensor,
             mean: bool = True,
             batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def drop_feature(x, drop_prob):
    drop_mask = th.empty((x.size(1), ),
                            dtype=th.float32,
                            device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x