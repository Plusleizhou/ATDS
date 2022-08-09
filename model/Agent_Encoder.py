import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from model.Layers import Res1d, Conv1d, LinearRes, Linear


class AgentEncoder(nn.Module):
    def __init__(self, config):
        super(AgentEncoder, self).__init__()
        self.config = config
        ng = 1

        n_in = 6
        n_out = [32, 64, 128]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        n = config["n_agent"]
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(n, n, ng=ng)
        if config["agent_extractor"] == 2:
            self.subgraph = Attention(n)
        else:
            self.subgraph = None

        self.input = nn.Sequential(
            nn.Linear(30, n),
            nn.ReLU(inplace=True),
            Linear(n, n, ng=ng, act=False),
        )

        self.ctrs = nn.Sequential(
            LinearRes(30, 32, ng=1),
            nn.Linear(32, 2)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, agents, agent_locs):
        out = agents

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        if self.subgraph is None:
            out = self.output(out)[:, :, -1]
        else:
            out = self.output(out).transpose(1, 2)
            out = self.subgraph(out)

        # agent_locs = rearrange(agent_locs[:, :, 10:].transpose(-1, -2), 'b c l -> b (c l)')
        agent_locs = rearrange(agent_locs[:, :, 10:], 'b c l -> b (c l)')
        agent_ctrs = self.ctrs(agent_locs)
        # out += self.input(agent_locs[:, -3:-1] + agent_ctrs)
        out += self.input(agent_locs)
        out = self.relu(out)
        return out, agent_ctrs


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_base = nn.Parameter(torch.arange(-3, 3, 6 / 20).unsqueeze(0).unsqueeze(0))
        self.attn = nn.Linear(dim, 1)
        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Linear(dim, dim)
        self.norm = nn.GroupNorm(1, dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        attn = self.attn(x).transpose(1, 2)
        attn = attn + self.attn_base
        attn = self.attend(attn)
        out = torch.matmul(attn, x)
        out = self.to_out(out)
        return self.norm(out[:, -1, :])
