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
        self.map_encoder = MapEncoder(config)

        self.a2m = A2M(config)
        self.m2a = M2A(config)
        self.a2a = A2A(config)

        self.pyramid_decoder = PyramidDecoder(config)

    def forward(self, data):
        # construct agent feature
        agents, agent_ids, agent_locs, agent_ctrs = agent_gather(gpu(data["trajs_obs"]), gpu(data["pad_obs"]))
        agents, d_agent_ctrs = self.agent_encoder(agents, agent_locs)
        agent_ctrs = get_agent_ctrs(d_agent_ctrs, agent_ctrs)

        # construct map features
        graph = graph_gather(to_long(gpu(data["graph"])))
        nodes, node_ids, node_ctrs = self.map_encoder(graph)

        # interactions
        nodes = self.a2m(nodes, graph, agents, agent_ids, agent_ctrs)
        agents = self.m2a(agents, agent_ids, agent_ctrs, nodes, node_ids, node_ctrs)
        agents = self.a2a(agents, agent_ids, agent_ctrs)

        # prediction
        out = self.pyramid_decoder(agents, agent_ids, agent_ctrs)
        rot, orig = gpu(data["rot"]), gpu(data["orig"])

        for i in range(len(out["reg"])):
            if "scaling_ratio" in data.keys():
                out["reg"][i] /= data["scaling_ratio"][i]
                out["key_points"][i] /= data["scaling_ratio"][i]
            if "flip" in data.keys() and data["flip"][i] < self.config["flip"]:
                out["reg"][i] = torch.cat((out["reg"][i][..., 1:], out["reg"][i][..., :1]), dim=-1)
                out['key_points'][i] = torch.cat((out["key_points"][i][..., 1:], out["key_points"][i][..., :1]), dim=-1)
            # out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(1, 1, 1, -1)
            # out['key_points'][i] = torch.matmul(out['key_points'][i], rot[i]) + orig[i].view(1, 1, 1, -1)
        return out


def get_agent_ctrs(d_agent_ctrs, agent_ctrs):
    count = 0
    for i in range(len(agent_ctrs)):
        agent_ctrs[i] += d_agent_ctrs[count: count + len(agent_ctrs[i])]
        count += len(agent_ctrs[i])
    return agent_ctrs


def agent_gather(trajs_obs, pad_obs):
    batch_size = len(trajs_obs)
    num_agents = [len(x) for x in trajs_obs]

    agents, agent_locs = [], []
    for i in range(batch_size):
        feats = torch.zeros_like(trajs_obs[i])
        feats[:, 1:, :] = trajs_obs[i][:, 1:, :] - trajs_obs[i][:, :-1, :]
        agents.append(torch.cat([feats, pad_obs[i].unsqueeze(2)], dim=-1))
        agent_locs.append(torch.cat([trajs_obs[i], pad_obs[i].unsqueeze(2)], dim=-1))

    agents = [x.transpose(1, 2) for x in agents]
    agents = torch.cat(agents, 0)

    agent_locs = [x.transpose(1, 2) for x in agent_locs]
    agent_locs = torch.cat(agent_locs, 0)

    agent_ids = []
    count = 0
    for i in range(batch_size):
        ids = torch.arange(count, count + num_agents[i]).to(agents.device)
        agent_ids.append(ids)
        count += num_agents[i]
    agent_ctrs = [agent_locs[ids, :2, -1] for ids in agent_ids]
    return agents, agent_ids, agent_locs, agent_ctrs


def graph_gather(graphs):
    batch_size = len(graphs)
    node_ids = []
    count = 0
    counts = []
    for i in range(batch_size):
        counts.append(count)
        ids = torch.arange(count, count + graphs[i]["num_nodes"]).to(
            graphs[i]["feats"].device
        )
        node_ids.append(ids)
        count = count + graphs[i]["num_nodes"]

    graph = dict()
    graph["ids"] = node_ids
    graph["ctrs"] = [x["ctrs"] for x in graphs]

    for key in ["feats", "turn", "control", "intersect"]:
        graph[key] = torch.cat([x[key] for x in graphs], 0)

    for k1 in ["pre", "suc"]:
        graph[k1] = []
        for i in range(len(graphs[0]["pre"])):
            graph[k1].append(dict())
            for k2 in ["u", "v"]:
                graph[k1][i][k2] = torch.cat(
                    [graphs[j][k1][i][k2] + counts[j] for j in range(batch_size)], 0
                )

    for k1 in ["left", "right"]:
        graph[k1] = dict()
        for k2 in ["u", "v"]:
            temp = [graphs[i][k1][k2] + counts[i] for i in range(batch_size)]
            temp = [
                x if x.dim() > 0 else graph["pre"][0]["u"].new().resize_(0)
                for x in temp
            ]
            graph[k1][k2] = torch.cat(temp)
    return graph


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)

    def forward(self, out, data):
        loss_out = self.pred_loss(out, gpu(data["trajs_fut"]), gpu(data["pad_fut"]))
        loss_out["loss"] = loss_out["cls_loss"] / (loss_out["num_cls"] + 1e-10) + \
                           loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10) + \
                           loss_out["key_points_loss"] / (loss_out["num_key_points"] + 1e-10)
        return loss_out
