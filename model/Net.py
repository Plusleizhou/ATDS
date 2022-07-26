import torch
from torch import nn
from utils import gpu, to_long
from model.Agent_Encoder import AgentEncoder
from model.Map_Encoder import MapEncoder
from model.Interaction_Encoder import A2M, M2A, A2A
from model.Pyramid_Decoder import PyramidDecoder
from model.Loss_Net import PredLoss as PredLoss


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config

        self.agent_encoder = AgentEncoder(config)
        # self.map_encoder = MapEncoder(config)

        # self.a2m = A2M(config)
        # self.m2a = M2A(config)
        self.a2a = A2A(config)

        self.pyramid_decoder = PyramidDecoder(config)

    def forward(self, trajs_obs, pad_obs, control, pre, right, suc, turn, intersect, ctrs, feats, left):
        # construct agent feature
        agents, agent_locs, agent_ctrs = agent_gather(gpu(trajs_obs), gpu(pad_obs))
        agents, d_agent_ctrs = self.agent_encoder(agents, agent_locs)
        agent_ctrs = get_agent_ctrs(d_agent_ctrs, agent_ctrs)

        # construct map features
        # nodes, node_ctrs = self.map_encoder(gpu(control), to_long(gpu(pre)), to_long(gpu(right)),
        #                                     to_long(gpu(suc)), gpu(turn), gpu(intersect), gpu(ctrs),
        #                                     gpu(feats), to_long(gpu(left)))

        # interactions
        # nodes = self.a2m(nodes, node_ctrs, agents, agent_ctrs)
        # agents = self.m2a(agents, agent_ctrs, nodes, node_ctrs)
        agents = self.a2a(agents, agent_ctrs)

        # prediction
        out = self.pyramid_decoder(agents, agent_ctrs)
        return out


def get_agent_ctrs(d_agent_ctrs, agent_ctrs):
    agent_ctrs = agent_ctrs + d_agent_ctrs
    return agent_ctrs


def agent_gather(trajs_obs, pad_obs):
    feats = torch.zeros_like(trajs_obs)
    feats[:, 1:, :] = trajs_obs[:, 1:, :] - trajs_obs[:, :-1, :]
    agents = torch.cat([feats, pad_obs.unsqueeze(2)], dim=-1)
    agent_locs = torch.cat([trajs_obs, pad_obs.unsqueeze(2)], dim=-1)

    agents = agents.transpose(1, 2)
    agents[:, :2] *= agents[:, -1:]
    agents[:, :2, 1:] *= agents[:, -1:, :-1]

    agent_locs = agent_locs.transpose(1, 2)
    agent_locs[:, :2] *= agent_locs[:, -1:]

    agent_ctrs = agent_locs[:, :2, -1]
    return agents, agent_locs, agent_ctrs


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)

    def forward(self, out, data):
        loss_out = self.pred_loss(out, gpu(data["trajs_fut"]), gpu(data["pad_fut"]), gpu(data["update_mask"]))
        loss_out["loss"] = loss_out["cls_loss"] / (loss_out["num_cls"] + 1e-10) + \
                           loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10) + \
                           loss_out["key_points_loss"] / (loss_out["num_key_points"] + 1e-10)
        return loss_out
