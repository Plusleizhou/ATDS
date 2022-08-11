import torch
from torch import nn
from model.Layers import Linear, index_add_naive
from einops import rearrange


class A2M(nn.Module):
    def __init__(self, config):
        super(A2M, self).__init__()
        self.config = config
        n_map = config["n_map"]

        att = []
        for i in range(2):
            att.append(Att(n_map, config["n_agent"]))
        self.att = nn.ModuleList(att)

    def forward(self, feat, agents, a2m):
        for i in range(len(self.att)):
            feat = self.att[i](
                feat,
                agents,
                a2m,
            )
        return feat


class M2A(nn.Module):
    def __init__(self, config):
        super(M2A, self).__init__()
        self.config = config

        n_agent = config["n_agent"]
        n_map = config["n_map"]

        att = []
        for i in range(2):
            att.append(Att(n_agent, n_map))
        self.att = nn.ModuleList(att)

    def forward(self, agents, nodes, m2a):
        for i in range(len(self.att)):
            agents = self.att[i](
                agents,
                nodes,
                m2a,
            )
        return agents


class A2A(nn.Module):
    def __init__(self, config):
        super(A2A, self).__init__()
        self.config = config

        n_agent = config["n_agent"]

        att = []
        for i in range(2):
            att.append(Att(n_agent, n_agent))
        self.att = nn.ModuleList(att)

    def forward(self, agents, a2a):
        for i in range(len(self.att)):
            agents = self.att[i](
                agents,
                agents,
                a2a,
            )
        return agents


class Att(nn.Module):
    def __init__(self, n_agt, n_ctx):
        super(Att, self).__init__()
        ng = 1
        self.n_heads = 6
        self.scale = n_ctx ** -0.5

        self.dist = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(inplace=True),
            Linear(2, 1, ng=ng, act=False),
        )

        self.to_q = Linear(n_agt, self.n_heads * n_ctx, ng=ng, act=False)
        self.to_k = Linear(n_agt, self.n_heads * n_ctx, ng=ng, act=False)
        self.to_v = Linear(n_agt, self.n_heads * n_ctx, ng=ng, act=True)
        self.sigmoid = nn.Sigmoid()

        self.agt = nn.Linear(n_agt, n_agt, bias=False)
        self.norm = nn.GroupNorm(ng, n_agt)
        self.linear = Linear(n_agt, n_agt, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)

        self.to_out = nn.Sequential(
            Linear(self.n_heads * n_ctx, n_agt, ng=ng),
            nn.Linear(n_agt, n_agt, bias=False),
        )

    def forward(self, agts, ctx, x2x):
        res = agts

        mask = x2x[:, :, 0]

        dist = x2x[:, :, 1:].reshape(-1, 2)
        dist = self.dist(dist)
        dist = dist.reshape(mask.shape[0], mask.shape[1], -1)

        q = self.relu(self.to_q(agts))
        k = self.relu(self.to_k(ctx))
        v = self.to_v(ctx)

        query, key, value = map(lambda t: rearrange(t, "n (h d) -> h n d", h=self.n_heads), [q, k, v])

        gates = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        gates = gates + dist[..., 0]
        gates = self.sigmoid(gates) * mask

        out = torch.matmul(gates, value)
        out = rearrange(out, "h n d -> n (h d)")
        out = self.to_out(out)

        agts = self.agt(agts)
        agts += out
        agts = self.norm(agts)
        agts = self.relu(agts)

        agts = self.linear(agts)
        agts += res
        agts = self.relu(agts)
        return agts
