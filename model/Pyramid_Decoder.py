import torch
import torch.nn as nn
from einops import rearrange
from model.Layers import Linear, LinearRes


class AttDest(nn.Module):
    def __init__(self, n_agt):
        super(AttDest, self).__init__()
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(inplace=True),
            Linear(n_agt, n_agt, ng=ng),
        )

        self.agt = Linear(2 * n_agt, n_agt, ng=ng)

    def forward(self, agts, agt_ctrs, dest_ctrs):
        dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).view(-1, 2)
        dist = self.dist(dist)
        agts = torch.cat((dist, agts), 1)
        agts = self.agt(agts)
        return agts


class TrajectoryDecoder(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(TrajectoryDecoder, self).__init__()
        self.out_dims = out_dims
        layers = []
        for i in range(3):
            layers.append(Linear(in_dims[i], in_dims[i + 1], ng=1))
            if i == 2:
                layers.append(nn.Sequential(
                    LinearRes(in_dims[i + 1], in_dims[i + 1], ng=1), nn.Linear(in_dims[i + 1], 2 * out_dims[i])))
            else:
                layers.append(nn.Sequential(
                    LinearRes(in_dims[i + 1] + in_dims[i + 2], in_dims[i + 1] + in_dims[i + 2], ng=1),
                    nn.Linear(in_dims[i + 1] + in_dims[i + 2], 2 * out_dims[i])))
            layers.append(Linear(2 * out_dims[i], in_dims[i + 1], ng=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, agents):
        pd_in, pd_out = [], []
        res = agents
        for i in range(3):
            res = self.layers[3 * i](res)
            pd_in.append(res)
        for i in range(3)[::-1]:
            key_points = self.layers[3 * i + 1](pd_in[i])
            if i != 0:
                key_point_features = self.layers[3 * i + 2](key_points)
                pd_in[i - 1] = torch.cat((pd_in[i - 1], key_point_features), dim=-1)
            key_points = rearrange(key_points, 'n (m1 m2 c) -> n m1 m2 c', m1=1, m2=self.out_dims[i], c=2)
            pd_out.append(key_points)
        return pd_out


class PyramidDecoder(nn.Module):
    def __init__(self, config):
        super(PyramidDecoder, self).__init__()
        self.config = config
        ng = 1

        n_agent = config["n_agent"]

        pred = []
        for i in range(config["num_mods"]):
            pred.append(TrajectoryDecoder([n_agent, 64, 32, 16], [30, 3, 1]))
        self.pred = nn.ModuleList(pred)

    def forward(self, agents, agent_ctrs):
        preds = self.pred[0](agents)
        reg = preds[-1]

        ctrs = agent_ctrs.view(-1, 1, 1, 2)
        reg = (reg + ctrs).squeeze(1)

        return reg


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config

    def forward(self, agents, agent_ids, agent_ctrs):
        speed = (agents.sum(-1) / (agents[:, -1].sum(-1) - 1 + 1e-10).unsqueeze(-1))
        agent_ctrs = torch.cat(agent_ctrs, dim=0)
        reg = agent_ctrs.view(-1, 1, 2).repeat(1, 30, 1)
        for i in range(self.config["num_preds"]):
            reg[:, i] = reg[:, i] + speed[:, :2] * (i + 1)

        reg = reg.unsqueeze(1).repeat(1, 6, 1, 1)
        reg[:, :, -1] += torch.randn_like(reg[:, :, -1])
        key_points = reg[:, :, [-1, 9, 19, -1]]
        cls = torch.softmax(torch.rand(reg.shape[0], reg.shape[1]), dim=-1)

        out = dict()
        out['cls'], out['reg'], out['key_points'] = [], [], []
        for i in range(len(agent_ids)):
            ids = agent_ids[i]
            out['cls'].append(cls[ids])
            out['reg'].append(reg[ids])
            out['key_points'].append(key_points[ids])
        return out
