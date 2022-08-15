import torch
from torch import nn
from model.Layers import Linear, LinearRes
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

    def forward(self, feat, graph, agents, agent_ids, agent_ctrs):
        for i in range(len(self.att)):
            feat = self.att[i](
                feat,
                graph["ids"],
                graph["ctrs"],
                agents,
                agent_ids,
                agent_ctrs,
                self.config["agent2map_dist"],
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

    def forward(self, agents, agent_ids, agent_ctrs, nodes,
                node_ids, node_ctrs):
        for i in range(len(self.att)):
            agents = self.att[i](
                agents,
                agent_ids,
                agent_ctrs,
                nodes,
                node_ids,
                node_ctrs,
                self.config["map2agent_dist"],
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

    def forward(self, agents, agent_ids, agent_ctrs):
        for i in range(len(self.att)):
            agents = self.att[i](
                agents,
                agent_ids,
                agent_ctrs,
                agents,
                agent_ids,
                agent_ctrs,
                self.config["agent2agent_dist"],
            )
        return agents


class Att(nn.Module):
    def __init__(self, n_agt, n_ctx):
        super(Att, self).__init__()
        ng = 1
        self.n_heads = 6
        self.scale = n_ctx ** -0.5

        self.dist = nn.Linear(2, 1)

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

    def forward(self, agts, agt_ids, agt_ctrs, ctx, ctx_ids, ctx_ctrs, dist_th):
        res = agts
        if len(ctx) == 0:
            agts = self.agt(agts)
            agts = self.relu(agts)
            agts = self.linear(agts)
            agts += res
            agts = self.relu(agts)
            return agts

        batch_size = len(agt_ids)
        hi, wi = [], []
        hi_count, wi_count = 0, 0
        for i in range(batch_size):
            dist = agt_ctrs[i].view(-1, 1, 2) - ctx_ctrs[i].view(1, -1, 2)
            dist = torch.sqrt((dist ** 2).sum(2))
            mask = dist <= dist_th

            ids = torch.nonzero(mask, as_tuple=False)
            if len(ids) == 0:
                continue

            hi.append(ids[:, 0] + hi_count)
            wi.append(ids[:, 1] + wi_count)
            hi_count += len(agt_ids[i])
            wi_count += len(ctx_ids[i])
        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)

        agt_ctrs = torch.cat(agt_ctrs, 0)
        ctx_ctrs = torch.cat(ctx_ctrs, 0)
        dist = agt_ctrs[hi] - ctx_ctrs[wi]
        dist = self.dist(dist)

        q = self.relu(self.to_q(agts[hi]))
        k = self.relu(self.to_k(ctx[wi]))
        v = self.to_v(ctx[wi])

        query, key, value = map(lambda t: rearrange(t, "n (h d) -> n h d", h=self.n_heads).unsqueeze(-2), [q, k, v])

        gates = torch.matmul(query, key.transpose(-1, -2)) * self.scale + dist.reshape(-1, 1, 1, 1)
        gates = self.sigmoid(gates)

        out = torch.matmul(gates, value).squeeze(-2)
        out = torch.zeros(agts.shape[0], self.n_heads, agts.shape[1]).to(agts.device).index_add(0, hi, out)
        out = rearrange(out, "n h d -> n (h d)")
        out = self.to_out(out)

        agts = self.agt(agts)
        agts = agts + out
        agts = self.norm(agts)
        agts = self.relu(agts)

        agts = self.linear(agts)
        agts += res
        agts = self.relu(agts)
        return agts
