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

SEED = 13
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_dummy_input():
    dataset = ProcessedDataset(config["processed_val"], mode="val")
    val_dataloader = DataLoader(dataset,
                                batch_size=1,
                                num_workers=config['val_workers'],
                                shuffle=False,
                                collate_fn=collate_fn)
    batch = val_dataloader.__iter__().next()
    dummy_input = list()
    tmp = list()
    for key in batch.keys():
        if key in ["trajs_obs", "pad_obs"]:
            dummy_input.append(batch[key][0])
        if key == "graph":
            graph = batch["graph"][0]

            dummy_input.append(graph["control"])

            for i in range(len(graph["pre"])):
                tmp.append((graph["pre"][i]["u"].type(torch.int64),
                            graph["pre"][i]["v"].type(torch.int64)))
            dummy_input.append(tuple(tmp))

            dummy_input.append((graph["right"]["u"].type(torch.int64),
                                graph["right"]["v"].type(torch.int64)))

            tmp = list()
            for i in range(len(graph["suc"])):
                tmp.append((graph["suc"][i]["u"].type(torch.int64),
                            graph["suc"][i]["v"].type(torch.int64)))
            dummy_input.append(tuple(tmp))

            dummy_input.append(graph["turn"])
            dummy_input.append(graph["intersect"])
            dummy_input.append(graph["ctrs"])
            dummy_input.append(graph["feats"])
            dummy_input.append((graph["left"]["u"].type(torch.int64),
                                graph["left"]["v"].type(torch.int64)))
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

    load_checkpoint(config["save_dir"] + "1.pth", model)
    model.eval()
    return model


def convert():
    model = load_model()
    input_names = ["trajs_obs", "pad_obs", "control",
                   "pre_0_u", "pre_0_v", "pre_1_u", "pre_1_v", "pre_2_u", "pre_2_v",
                   "pre_3_u", "pre_3_v", "pre_4_u", "pre_4_v", "pre_5_u", "pre_5_v",
                   "right_u", "right_v",
                   "suc_0_u", "suc_0_v", "suc_1_u", "suc_1_v", "suc_2_u", "suc_2_v",
                   "suc_3_u", "suc_3_v", "suc_4_u", "suc_4_v", "suc_5_u", "suc_5_v",
                   "turn", "intersect",
                   "ctrs", "feats",
                   "left_u", "left_v"]
    output_names = ["cls", "reg", "key_points"]

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
    print(f"cls: {out['cls'].shape}, reg: {out['reg'].shape}, key_points: {out['key_points'].shape}")
    print(out['cls'][0])
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
        "trajs_obs": dummy_input[0],
        "pad_obs": dummy_input[1],
        "control": dummy_input[2],
        "pre_0_u": dummy_input[3][0][0],
        "pre_0_v": dummy_input[3][0][1],
        "pre_1_u": dummy_input[3][1][0],
        "pre_1_v": dummy_input[3][1][1],
        "pre_2_u": dummy_input[3][2][0],
        "pre_2_v": dummy_input[3][2][1],
        "pre_3_u": dummy_input[3][3][0],
        "pre_3_v": dummy_input[3][3][1],
        "pre_4_u": dummy_input[3][4][0],
        "pre_4_v": dummy_input[3][4][1],
        "pre_5_u": dummy_input[3][5][0],
        "pre_5_v": dummy_input[3][5][1],
        "right_u": dummy_input[4][0],
        "right_v": dummy_input[4][1],
        "suc_0_u": dummy_input[5][0][0],
        "suc_0_v": dummy_input[5][0][1],
        "suc_1_u": dummy_input[5][1][0],
        "suc_1_v": dummy_input[5][1][1],
        "suc_2_u": dummy_input[5][2][0],
        "suc_2_v": dummy_input[5][2][1],
        "suc_3_u": dummy_input[5][3][0],
        "suc_3_v": dummy_input[5][3][1],
        "suc_4_u": dummy_input[5][4][0],
        "suc_4_v": dummy_input[5][4][1],
        "suc_5_u": dummy_input[5][5][0],
        "suc_5_v": dummy_input[5][5][1],
        "turn": dummy_input[6],
        "intersect": dummy_input[7],
        "ctrs": dummy_input[8],
        "feats": dummy_input[9],
        "left_u": dummy_input[10][0],
        "left_v": dummy_input[10][1],
    }
    start_time = time.time()
    for i in range(100):
        ort_outs = ort_session.run(None, inputs)
    print("onnx model results".center(50, "-"))
    print(f"Average time for inference once: {round((time.time() - start_time) * 10, 3)} ms")
    print([x.shape for x in ort_outs])
    print(ort_outs[0][0])

    print("torch model results".center(50, "-"))
    torch_out = run_torch()
    np.testing.assert_allclose(to_numpy(torch_out["cls"]), ort_outs[0], rtol=1e-03, atol=1e-05)


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
