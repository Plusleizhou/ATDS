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

    def forward(self, feat, ctrs, agents, agent_ctrs, a2m):
        for i in range(len(self.att)):
            feat = self.att[i](
                feat,
                ctrs,
                agents,
                agent_ctrs,
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

    def forward(self, agents, agent_ctrs, nodes, node_ctrs, m2a):
        for i in range(len(self.att)):
            agents = self.att[i](
                agents,
                agent_ctrs,
                nodes,
                node_ctrs,
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

    def forward(self, agents, agent_ctrs, a2a):
        for i in range(len(self.att)):
            agents = self.att[i](
                agents,
                agent_ctrs,
                agents,
                agent_ctrs,
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
            nn.Linear(2, n_ctx),
            nn.ReLU(inplace=True),
            Linear(n_ctx, n_ctx, ng=ng),
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

    def forward(self, agts, agt_ctrs, ctx, ctx_ctrs, x2x):
        res = agts

        hi = x2x[0]
        wi = x2x[1]

        dist = agt_ctrs[hi] - ctx_ctrs[wi]
        dist = self.dist(dist)

        q = self.relu(self.to_q(agts[hi] + dist))
        k = self.relu(self.to_k(ctx[wi] + dist))
        v = self.to_v(ctx[wi])

        query, key, value = map(lambda t: rearrange(t, "n (h d) -> n h d", h=self.n_heads).unsqueeze(-2), [q, k, v])

        gates = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        gates = self.sigmoid(gates)

        out = torch.matmul(gates, value).squeeze(-2)
        out = rearrange(out, "n h d -> n (h d)")
        out = self.to_out(out)

        agts = self.agt(agts)
        # agts.index_add_(0, hi, out)
        # agts.scatter_add_(0, hi.unsqueeze(1).repeat(1, 128), out)
        agts = index_add_naive(agts, out, hi)
        agts = self.norm(agts)
        agts = self.relu(agts)

        agts = self.linear(agts)
        agts += res
        agts = self.relu(agts)
        return agts
