from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import tools.base_utils
from load_config import config
from model.Net import Net
from model.Net import Loss
from processed_data import collate_fn
import os
from train import load_prev_weights
from data.data_utils.data_sampling import SamplingDataset
from eval import loader
from matplotlib import pyplot as plt
import argparse
import shutil
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ATDS-Net Evaluating')
    parser.add_argument('--save_path', default='log-0', type=str, help='checkpoint path')
    parser.add_argument("--val_path", default="/home/plusai/code_space/dataset/features/val")
    parser.add_argument("--seq_id", default=0, type=int, help="the number index of the sorted file list")
    parser.add_argument('--devices', default='0', type=str, help='gpu devices for training')
    parser.add_argument("--batch_size", default=2, type=int, help='frame interval')
    parser.add_argument('--mode', default='ego', type=str, help='ego/seq/base')
    args = parser.parse_args()

    # update config
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    model_name = args.save_path
    config["save_dir"] = os.path.join("results", model_name, "weights/")
    config["result"] = os.path.join("results", model_name, "result.txt")
    config["images"] = os.path.join("results", model_name, "images/")
    config["competition_files"] = os.path.join("results", model_name, "competition/")
    config["val_batch_size"] = args.batch_size

    # get val path
    dst_path = os.path.join(os.path.dirname(args.val_path), "../", str(args.seq_id))
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    if len(os.listdir(dst_path)) == 0:
        sd = SamplingDataset(args.val_path, dst_path)
        for file in sd.sorted_file_list:
            bag_name = sd.get_bag_name_and_frame(file)[0]
            if bag_name == list(sd.bag_dict.keys())[args.seq_id]:
                src = os.path.join(args.val_path, file)
                dst = os.path.join(dst_path, file)
                shutil.copyfile(src, dst)
    config["processed_val"] = dst_path
    return args


def visualization(num, ax, out, data):
    plt.cla()
    ax.set_title(data["seq_id"][0], fontweight="bold")
    ax.axis("equal")

    data["trajs_obs"] = [x[:, :, :2] for x in data["trajs_obs"]]

    orig = data['orig'][0].detach().cpu().numpy()
    rot = data["rot"][0].detach().cpu().numpy()
    x_min = np.min(orig[0]) - 100
    x_max = np.max(orig[0]) + 100
    y_min = np.min(orig[1]) - 100
    y_max = np.max(orig[1]) + 100
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    color_map = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    colors = ['#2A2A2A', '#481184', '#083D7E', '#005924', '#8E2D03', '#8B2B03',
              '#762905', '#772A05', '#8F0000', '#7A002D', '#5C006F', '#650762',
              '#084D8F', '#02446B', '#142772', '#015947', '#005622', '#004E2C']

    # map
    ctrs = data["graph"][0]["ctrs"].detach().cpu().numpy()
    ctrs[:, :2] = ctrs[:, :2].dot(rot.T) + orig

    feats = data["graph"][0]["feats"].detach().cpu().numpy()
    feats[:, :2] = feats[:, :2].dot(rot.T)
    for j in range(feats.shape[0]):
        vec = feats[j]
        pt0 = ctrs[j] - vec / 2
        pt1 = ctrs[j] + vec / 2
        ax.arrow(pt0[0], pt0[1], (pt1 - pt0)[0], (pt1 - pt0)[1], edgecolor=None, color="grey", alpha=0.5)

    # ego
    gt_preds = data["trajs_fut"][0][0].detach().cpu().numpy().dot(rot.T) + orig
    pad_fut = data["pad_fut"][0][0].detach().cpu().numpy()
    ax.plot(gt_preds[:-1, 0], gt_preds[:-1, 1], marker=".", alpha=1, color="g", zorder=20)
    ax.arrow(gt_preds[-2, 0], gt_preds[-2, 1], (gt_preds[-1] - gt_preds[-2])[0], (gt_preds[-1] - gt_preds[-2])[1],
             color="g", alpha=1, width=1)

    trajs_obs = data["trajs_obs"][0][0].detach().cpu().numpy().dot(rot.T) + orig
    pad_obs = data["pad_obs"][0][0].detach().cpu().numpy()
    ax.plot(trajs_obs[:, 0], trajs_obs[:, 1], marker=".", alpha=1, color="r", zorder=20)

    all_points = np.expand_dims(np.concatenate([trajs_obs, gt_preds], axis=0), 0)

    # surrounding agents
    has_preds = data["has_preds"][0].detach().cpu().numpy().astype(np.bool)
    pred_trajs = data["pred_trajs"][0].detach().cpu().numpy().dot(rot.T) + orig
    gt_preds = data["trajs_fut"][0][has_preds].detach().cpu().numpy().dot(rot.T) + orig
    pad_fut = data["pad_fut"][0][has_preds].detach().cpu().numpy() == 1
    trajs_obs = data["trajs_obs"][0][has_preds].detach().cpu().numpy().dot(rot.T) + orig
    pad_obs = data["pad_obs"][0][has_preds].detach().cpu().numpy().astype(np.bool)
    preds = out['reg'][0][has_preds].detach().cpu().numpy().dot(rot.T) + orig
    probabilities = out['cls'][0][has_preds].detach().cpu().numpy() * 100
    for i in range(gt_preds.shape[0]):
        gt_pred = gt_preds[i][pad_fut[i]]
        if gt_pred.shape[0] == 0:
            continue
        traj_obs = trajs_obs[i][pad_obs[i]]
        pred_traj = pred_trajs[i, :pad_fut[i].shape[0]][pad_fut[i]]
        ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1], marker="X", c="orange", s=40, zorder=20, alpha=1)
        ax.plot(gt_pred[:-1, 0], gt_pred[:-1, 1],
                marker=".", alpha=1, color=colors[i % 18], zorder=20)
        if gt_pred.shape[0] >= 2:
            ax.arrow(gt_pred[-2, 0], gt_pred[-2, 1],
                     (gt_pred[-1] - gt_pred[-2])[0], (gt_pred[-1] - gt_pred[-2])[1],
                     color=colors[i % 18], alpha=1, width=1)
        else:
            ax.arrow(traj_obs[-1, 0], traj_obs[-1, 1],
                     (gt_pred[-1] - traj_obs[-1, 0])[0], (gt_pred[-1] - traj_obs[-1, 1])[1],
                     color=colors[i % 18], alpha=1, width=1)
        ax.plot(traj_obs[:, 0], traj_obs[:, 1], marker=".", alpha=0.5, color=colors[i % 18], zorder=20)
        ax.scatter(traj_obs[:, 0], traj_obs[:, 1], marker="o",
                   c=np.arange(traj_obs.shape[0]), cmap=color_map[i % 18])
        for k in range(preds.shape[1]):
            pred = preds[i, k][pad_fut[i]]
            ax.scatter(pred[-1, 0], pred[-1, 1], marker="X", c="orange", s=10, zorder=20, alpha=1)
            ax.text(pred[-1, 0], pred[-1, 1], str("%d" % probabilities[i, k]), fontsize=8, alpha=0.5,
                    horizontalalignment="center", verticalalignment="bottom")

    all_points = np.concatenate([all_points, np.concatenate([trajs_obs, gt_preds], axis=1)], axis=0)
    x_min = np.min(all_points[..., 0]) - 20
    x_max = np.max(all_points[..., 0]) + 20
    y_min = np.min(all_points[..., 1]) - 20
    y_max = np.max(all_points[..., 1]) + 20
    center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
    width = np.max([x_max - x_min, y_max - y_min])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    base_ade1, base_fde1, ade1, fde1, ade, fde, min_ids, _ = tools.base_utils.pred_metrics(
        preds, pred_trajs, gt_preds, pad_fut, config["num_preds"])
    ax.text(center[0] + width / 2, center[1] - width / 2,
            "b_ade1:{:.3f}\nb_fde1:{:.3f}\nade1: {:.3f}\nfde1: {:.3f}\nade6: {:.3f}\nfde6: {:.3f}".
            format(base_ade1, base_fde1, ade1, fde1, ade, fde),
            fontsize=15, horizontalalignment="right", verticalalignment="bottom")

    preds = preds[pad_fut[:, config["num_preds"] - 1]]
    for i, idx in enumerate(min_ids):
        ax.scatter(preds[i, idx, -1, 0], preds[i, idx, -1, 1], marker="X", c="b", s=40, zorder=20, alpha=1)
        ax.text(preds[i, idx, -1, 0], preds[i, idx, -1, 1], str("%d" % probabilities[i, idx]), fontsize=8, alpha=1,
                horizontalalignment="center", verticalalignment="bottom")

    # ax.legend(loc="upper right", shadow=True)
    plt.pause(0.01)
    return ax,


def main():
    args = parse_args()
    _, ProcessedDataset, PostProcess = loader(args)
    dataset = ProcessedDataset(config["processed_val"], mode="val")
    val_dataloader = DataLoader(dataset,
                                batch_size=config["val_batch_size"],
                                num_workers=config['val_workers'],
                                shuffle=False,
                                collate_fn=collate_fn)
    net = Net(config).cuda()
    loss_net = Loss(config).cuda()
    net.eval()

    if len(os.listdir(config['save_dir'])) > 0:
        files = os.listdir(config['save_dir'])
        if 'net.pth' in files:
            files.remove('net.pth')
        files.sort(key=lambda x: float(x[:-4]))
        path = config['save_dir'] + files[-1]
        load_prev_weights(net, path)
        # net.load_state_dict(torch.load(path, map_location="cuda:0"))
        print('load weights from %s' % path)
    else:
        print('from beginning!')
        # exit()

    metrics = dict()
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.ion()
    loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Val", leave=True)
    loop.set_postfix_str(f'total_loss="???", minFDE="???"')
    for i, batch in loop:
        with torch.no_grad():
            out = net(batch)
            loss_out = loss_net(out, batch)

            post_out = PostProcess(out, batch)  # 只记录了agent的pred，gt_pred, 和has_pred
            post_out.append(metrics, loss_out)  # 加入了全部agent的loss_out

            visualization(i, ax, out, batch)

    plt.ioff()
    plt.show()
    val_out = post_out.display(metrics)
    for k, v in val_out.items():
        print("{}: {}".format(k, v))


if __name__ == '__main__':
    main()
