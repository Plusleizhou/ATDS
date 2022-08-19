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
        #
        # self.a2m = A2M(config)
        # self.m2a = M2A(config)
        self.a2a = A2A(config)

        self.pyramid_decoder = PyramidDecoder(config)

    def forward(self, agents, nodes, map_indexes, a2m, m2a, a2a):
        # extract useful info
        # agents = agents
        # nodes = nodes
        # construct agent feature
        agent_ctrs = agents[:, 6:8, 19]
        agents, d_agent_ctrs = self.agent_encoder(agents[:, :6], agents[:, 6:])
        agent_ctrs = get_agent_ctrs(d_agent_ctrs, agent_ctrs)

        # # construct map features
        # nodes, node_ctrs = self.map_encoder(nodes, map_indexes)
        #
        # # interactions
        # nodes = self.a2m(nodes, agents, a2m)
        # agents = self.m2a(agents, nodes, m2a)
        agents = self.a2a(agents, a2a)

        # prediction
        out = self.pyramid_decoder(agents, agent_ctrs)
        return out


def get_agent_ctrs(d_agent_ctrs, agent_ctrs):
    agent_ctrs = agent_ctrs + d_agent_ctrs
    return agent_ctrs


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
