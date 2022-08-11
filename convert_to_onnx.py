import os
import argparse
import torch
import numpy as np
import time
import onnx
import onnxruntime

from onnxsim import simplify
from processed_data import ProcessedDataset, collate_fn
from torch.utils.data import DataLoader
from model.Net import Net
from load_config import config
from utils import gpu

SEED = 13
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_dummy_input():
    def get_interaction_indexes(agt_ctrs, ctx_ctrs, dist_th):
        dist = agt_ctrs.view(-1, 1, 2) - ctx_ctrs.view(1, -1, 2)
        dist = torch.sqrt((dist ** 2).sum(2))
        roi = dist <= dist_th
        return torch.nonzero(roi, as_tuple=False)

    def agent_gather(trajs_obs, pad_obs):
        feats = torch.zeros_like(trajs_obs[:, :, :2])
        feats[:, 1:, :] = trajs_obs[:, 1:, :2] - trajs_obs[:, :-1, :2]
        agts = torch.cat([feats, trajs_obs[:, :, 2:4] / 10.0, trajs_obs[:, :, 4:],
                          pad_obs.unsqueeze(2)], dim=-1)
        agt_locs = torch.cat([trajs_obs[:, :, :2], pad_obs.unsqueeze(2)], dim=-1)

        agts = agts.transpose(1, 2)
        agts[:, :-1] *= agts[:, -1:]
        agts[:, :-1, 1:] *= agts[:, -1:, :-1]

        agt_locs = agt_locs.transpose(1, 2)
        agt_locs[:, :2] *= agt_locs[:, -1:]

        agts = torch.cat([agts, agt_locs], dim=1)
        return agts

    dataset = ProcessedDataset("./data/features/benchmark/", mode="val")
    val_dataloader = DataLoader(dataset,
                                batch_size=1,
                                num_workers=config['val_workers'],
                                shuffle=False,
                                collate_fn=collate_fn)
    val_dataloader = val_dataloader.__iter__()
    for i in range(0):
        batch = val_dataloader.next()
    batch = val_dataloader.next()
    dummy_input = list()

    # max num of nodes
    agents_num = 32
    nodes_num = 1000

    # agents
    agents_pad = torch.zeros(agents_num, 9, 20)
    agents = agent_gather(batch["trajs_obs"][0], batch["pad_obs"][0])
    agents_pad[:agents.shape[0]] = agents
    dummy_input.append(agents_pad)

    # hd_maps
    graph = batch["graph"][0]

    nodes_pad = torch.zeros(nodes_num, 8)
    nodes = torch.cat(
            (
                graph["ctrs"],
                graph["feats"],
                graph["turn"],
                graph["control"].unsqueeze(1),
                graph["intersect"].unsqueeze(1),
            ),
            1,
    )
    nodes_pad[:nodes.shape[0]] = nodes
    dummy_input.append(nodes_pad)

    # map_indexes = torch.arange(nodes_pad.shape[0], dtype=torch.int64).unsqueeze(1).repeat(1, 28)
    map_indexes = torch.ones(nodes_pad.shape[0], 28, dtype=torch.int64) * (nodes_num - 1)
    for i in range(len(graph["pre"])):
        map_indexes[:graph["pre"][i]["u"].shape[0], 2 * i] = graph["pre"][i]["u"].type(torch.int64)
        map_indexes[:graph["pre"][i]["v"].shape[0], 2 * i + 1] = graph["pre"][i]["v"].type(torch.int64)

    map_indexes[:graph["right"]["u"].shape[0], 12] = graph["right"]["u"].type(torch.int64)
    map_indexes[:graph["right"]["v"].shape[0], 13] = graph["right"]["v"].type(torch.int64)

    for i in range(len(graph["suc"])):
        map_indexes[:graph["suc"][i]["u"].shape[0], 14 + 2 * i] = graph["suc"][i]["u"].type(torch.int64)
        map_indexes[:graph["suc"][i]["v"].shape[0], 15 + 2 * i] = graph["suc"][i]["v"].type(torch.int64)

    map_indexes[:graph["left"]["u"].shape[0], 26] = graph["left"]["u"].type(torch.int64)
    map_indexes[:graph["left"]["v"].shape[0], 27] = graph["left"]["v"].type(torch.int64)
    dummy_input.append(map_indexes)

    action_indexes = torch.ones(nodes_pad.shape[0], 6, dtype=torch.int64)
    # action_indexes[:, [0, 3]] = torch.arange(nodes_pad.shape[0], dtype=torch.int64).unsqueeze(1).repeat(1, 2)
    action_indexes[:, [0, 3]] *= (nodes_num - 1)
    action_indexes[:, [1, 2, 4, 5]] *= (agents_num - 1)
    agent_ctrs = batch["trajs_obs"][0][:, -1, :2]
    node_ctrs = graph["ctrs"][:, :2]
    ids = get_interaction_indexes(node_ctrs, agent_ctrs, config["agent2map_dist"])
    action_indexes[:ids.shape[0], :2] = ids
    ids = get_interaction_indexes(agent_ctrs, node_ctrs, config["map2agent_dist"])
    action_indexes[:ids.shape[0], 2:4] = ids
    ids = get_interaction_indexes(agent_ctrs, agent_ctrs, config["agent2agent_dist"])
    action_indexes[:ids.shape[0], 4:] = ids
    dummy_input.append(action_indexes)

    dummy_input = gpu(dummy_input)
    return tuple(dummy_input)


def load_checkpoint(path, net):
    pre_state = torch.load(path, map_location="cpu")
    state_dict = net.state_dict()
    loaded_modules = []
    for k, v in pre_state.items():
        k = k.replace('module.', '', 1)
        if k in state_dict.keys() and state_dict[k].shape == v.shape:
            state_dict[k] = v
            loaded_modules.append(k)
    net.load_state_dict(state_dict)
    print(f'loaded parameters {len(loaded_modules)}/{len(state_dict)}')


def load_model():
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(config)
    model = model.to(device=device)

    load_checkpoint(config["save_dir"] + "2.pth", model)
    model.eval()
    return model


def convert():
    model = load_model()
    input_names = ["agents", "nodes", "map_indexes", "action_indexes"]
    output_names = ["reg", "key_points"]

    torch.onnx.export(model, get_dummy_input(), "atdsnet.onnx", verbose=True, opset_version=11, input_names=input_names,
                      output_names=output_names)


def simplify_onnx():
    # Load the ONNX model
    model = onnx.load("atdsnet.onnx")
    # Check that the model is well-formed
    onnx.checker.check_model(model)
    # Print a human-readable representation of the graph
    print('model graph = ', onnx.helper.printable_graph(model.graph))

    # convert model
    model_simp, check = simplify(model)

    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, 'simple_atdsnet.onnx')


def run_torch():
    model = load_model()

    dummy_input = get_dummy_input()

    out = model(*dummy_input)
    # print(model)
    print(f"reg: {out['reg'].shape}, key_points: {out['key_points'].shape}")
    print(out['key_points'][0, 0, 0])
    return out


def run_onnx():
    ort_session = onnxruntime.InferenceSession("atdsnet.onnx", providers=['TensorrtExecutionProvider',
                                                                                 'CUDAExecutionProvider',
                                                                                 'CPUExecutionProvider'])

    # compute ONNX Runtime output prediction
    def to_numpy(data):
        if isinstance(data, dict):
            for key in data.keys():
                data[key] = to_numpy(data[key])
        if isinstance(data, list) or isinstance(data, tuple):
            data = [to_numpy(x) for x in data]
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        return data

    dummy_input = to_numpy(get_dummy_input())
    inputs = {
        "agents": dummy_input[0],
        "nodes": dummy_input[1],
        "map_indexes": dummy_input[2],
        "action_indexes": dummy_input[3],
    }
    start_time = time.time()
    for i in range(100):
        ort_outs = ort_session.run(None, inputs)
    print("onnx model results".center(50, "-"))
    print(f"Average time for inference once: {round((time.time() - start_time) * 10, 3)} ms")
    print([x.shape for x in ort_outs])
    print(ort_outs[1][0, 0, 0])

    print("torch model results".center(50, "-"))
    torch_out = run_torch()
    np.testing.assert_allclose(to_numpy(torch_out["key_points"]), ort_outs[1], rtol=1e-03, atol=1e-05)


def get_args():
    parser = argparse.ArgumentParser(description='Convert pytorch model to onnx model')
    parser.add_argument('--to_onnx', action="store_true", default=False,
                        help="whether to convert the pytorch model to onnx model")
    parser.add_argument("--simplify", action="store_true", default=False,
                        help="whether the simplify the onnx model")
    parser.add_argument("--run", action="store_true", default=False, help="whether to use the model")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.to_onnx:
        convert()
    if args.simplify:
        assert "atdsnet.onnx" in os.listdir(os.getcwd()), "convert the pytorch model to onnx model first"
        simplify_onnx()
    if args.run:
        run_onnx()
    run_torch()
