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
    return batch


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
    input_names = ["input"]
    output_names = ["output"]

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

    out = model(dummy_input)
    # print(model)
    print(f"cls: {out['cls'][0].shape}, reg: {out['reg'][0].shape}, key_points: {out['key_points'][0].shape}")
    print(out['cls'][0][0])
    return out


def run_onnx():
    ort_session = onnxruntime.InferenceSession("simple_atdsnet.onnx", providers=['TensorrtExecutionProvider',
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
    graph = dummy_input["graph"][0]
    inputs = {"x.1": dummy_input["trajs_obs"][0], "x.3": dummy_input["pad_obs"][0], "_data.1": graph["control"],
              "_data.5": graph["pre"][0]["u"], "_data.9": graph["pre"][0]["v"],
              "_data.13": graph["pre"][1]["u"], "_data.17": graph["pre"][1]["v"],
              "_data.21": graph["pre"][2]["u"], "_data.25": graph["pre"][2]["v"],
              "_data.29": graph["pre"][3]["u"], "_data.33": graph["pre"][3]["v"],
              "_data.37": graph["pre"][4]["u"], "_data.41": graph["pre"][4]["v"],
              "_data.45": graph["pre"][5]["u"], "_data.49": graph["pre"][5]["v"],
              "_data.53": graph["right"]["u"], "_data.57": graph["right"]["v"],
              "_data.61": graph["suc"][0]["u"], "_data.65": graph["suc"][0]["v"],
              "_data.69": graph["suc"][1]["u"], "_data.73": graph["suc"][1]["v"],
              "_data.77": graph["suc"][2]["u"], "_data.81": graph["suc"][2]["v"],
              "_data.85": graph["suc"][3]["u"], "_data.89": graph["suc"][3]["v"],
              "_data.93": graph["suc"][4]["u"], "_data.97": graph["suc"][4]["v"],
              "_data.101": graph["suc"][5]["u"], "_data.105": graph["suc"][5]["v"],
              "_data.121": graph["turn"], "_data.125": graph["intersect"],
              "_data.129": graph["ctrs"], "_data.133": graph["feats"],
              "_data.137": graph["left"]["u"], "_data.141": graph["right"]["v"],
              }
    start_time = time.time()
    for i in range(100):
        ort_outs = ort_session.run(None, inputs)
    print(f"Average time for inference once: {round((time.time() - start_time) * 10, 3)} ms")
    print([x.shape for x in ort_outs])
    print(ort_outs[0][0])

    torch_out = run_torch()
    np.testing.assert_allclose(to_numpy(torch_out["cls"][0]), ort_outs[0], rtol=1e-03, atol=1e-05)


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
