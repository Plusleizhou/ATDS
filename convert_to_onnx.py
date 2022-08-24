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


def get_dummy_input(args):
    # max num of nodes
    agents_num = args.agents

    # agents
    dummy_input = torch.ones(agents_num, 20 * 9 + agents_num * 3)
    return dummy_input


def load_input(args):
    def get_interaction_indexes(agt_ctrs, ctx_ctrs, dist_th):
        dist = agt_ctrs.view(-1, 1, 2) - ctx_ctrs.view(1, -1, 2)
        l2_dist = torch.sqrt((dist ** 2).sum(2))
        roi = l2_dist <= dist_th
        return torch.cat([roi.unsqueeze(2), dist], dim=2)

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
    for i in range(100):
        val_dataloader.next()
    batch = val_dataloader.next()

    # max num of nodes
    agents_num = args.agents

    # agents
    agents_pad = torch.zeros(agents_num, 20, 9)
    agents = agent_gather(batch["trajs_obs"][0], batch["pad_obs"][0]).transpose(1, 2)
    agents_pad[:agents.shape[0]] = agents

    agent_ctrs = batch["trajs_obs"][0][:, -1, :2]
    a2a_pad = torch.zeros(agents_num, agents_num, 3, dtype=torch.float32)
    a2a = get_interaction_indexes(agent_ctrs, agent_ctrs, config["agent2agent_dist"])
    a2a_pad[:a2a.shape[0], :a2a.shape[1]] = a2a

    dummy_input = torch.cat([agents_pad.view(agents_num, -1), a2a_pad.view(agents_num, -1)], dim=-1)
    return gpu(dummy_input)


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


def convert(args):
    model = load_model()
    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(model, load_input(args), "atdsnet.onnx", verbose=True, opset_version=11,
                      input_names=input_names, output_names=output_names)


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


def run_torch(args, multi=False):
    model = load_model()

    dummy_input = load_input(args)

    if multi:
        start_time = time.time()
        for i in range(100):
            _ = model(dummy_input)
        print(f"Average time for inference once: {round((time.time() - start_time) * 10, 3)} ms")

    out = model(dummy_input)
    # print(model)
    print(f"reg: {out.shape}")
    print(out[0, 10:].detach().cpu().numpy())
    return out


def run_onnx(args):
    ort_session = onnxruntime.InferenceSession("atdsnet.onnx", providers=['CUDAExecutionProvider',
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

    dummy_input = to_numpy(load_input(args))
    inputs = {
        "input": dummy_input,
    }
    print("onnx model results".center(50, "-"))
    start_time = time.time()
    for i in range(100):
        ort_outs = ort_session.run(None, inputs)
    print(f"Average time for inference once: {round((time.time() - start_time) * 10, 3)} ms")
    print(ort_outs[0].shape)
    print(ort_outs[0][0, 10:])

    print("torch model results".center(50, "-"))
    torch_out = run_torch(args, multi=True)
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)


def get_args():
    parser = argparse.ArgumentParser(description='Convert pytorch model to onnx model')
    parser.add_argument('--to_onnx', action="store_true", default=False,
                        help="whether to convert the pytorch model to onnx model")
    parser.add_argument("--simplify", action="store_true", default=False,
                        help="whether the simplify the onnx model")
    parser.add_argument("--run", action="store_true", default=False, help="whether to use the model")
    parser.add_argument("--run_torch", action="store_true", default=False, help="run torch model only")
    parser.add_argument("--agents", type=int, default=16, help="the maximin num of agents")
    parser.add_argument("--lane_nodes", type=int, default=500, help="the maximin num of lane nodes")
    return parser.parse_args()


def main(args):
    if args.to_onnx:
        convert(args)
    if args.simplify:
        assert "atdsnet.onnx" in os.listdir(os.getcwd()), "convert the pytorch model to onnx model first"
        simplify_onnx()
    if args.run:
        run_onnx(args)
    if args.run_torch:
        run_torch(args)


if __name__ == "__main__":
    main(get_args())

