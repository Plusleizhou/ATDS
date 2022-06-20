from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from load_config import config
import time
from model.Net import Net
from utils import PostProcess, gpu
from processed_data import ProcessedDataset, collate_fn
from utils import visualization
from train import load_prev_weights

import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ATDS-Net Evaluating')
    parser.add_argument('--save_path', default='log-0', type=str, help='checkpoint path')
    parser.add_argument("--use_goal", action="store_true", default=False, help="whether to use goal")
    parser.add_argument('--devices', default='0', type=str, help='gpu devices for training')
    args = parser.parse_args()

    # update config
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    model_name = args.save_path
    config["save_dir"] = os.path.join("results", model_name, "weights/")
    config["result"] = os.path.join("results", model_name, "result.txt")
    config["images"] = os.path.join("results", model_name, "images/")
    config["competition_files"] = os.path.join("results", model_name, "competition/")
    return args


def main():
    args = parse_args()
    dataset = ProcessedDataset(config["processed_test"], mode="test")
    test_dataloader = DataLoader(dataset,
                                 batch_size=config['test_batch_size'],
                                 shuffle=True,
                                 collate_fn=collate_fn)
    net = Net(config).cuda()
    net.eval()

    if len(os.listdir(config['save_dir'])) > 0:
        files = os.listdir(config['save_dir'])
        if 'net.pth' in files:
            files.remove('net.pth')
        files.sort(key=lambda x: float(x[:-4]))
        path = config['save_dir'] + files[-1]
        load_prev_weights(net, path)
        # net.load_state_dict(torch.load(path))
        print('load weights from %s' % path)
    else:
        print('from beginning!')
        exit()

    loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Test", leave=True)
    for i, batch in loop:
        with torch.no_grad():
            out = net(batch)
            if args.use_goal:
                for k in range(len(out['reg'])):
                    out['reg'][k][:, :, -1] = out['key_points'][k][:, :, -1]
            if (i + 1) % 100 == 0:
                visualization(out, batch, i, True, False)


if __name__ == '__main__':
    main()
