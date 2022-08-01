import numpy as np
import torch
import os
from load_config import config
import matplotlib.pyplot as plt


def gpu(data):
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key: gpu(_data) for key, _data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data


def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data


class PostProcess(object):
    def __init__(self, out=None, data=None):
        # agent trajectory is in the first pos
        self.post_out = dict()
        if out is not None:
            self.post_out['preds'] = [x[0:1].detach().cpu().numpy() for x in out['reg']]
        if data is not None:
            self.post_out["gt_preds"] = [x[0:1].numpy() for x in data["trajs_fut"]]
            self.post_out["has_preds"] = [x[0:1].numpy() for x in data["pad_fut"]]

    def append(self, metrics, loss_out):
        # initialize
        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != 'loss':
                    metrics[key] = 0

            for key in self.post_out:
                metrics[key] = []

        # convert to scalar in python
        for key in loss_out:
            if key == 'loss':
                continue
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

        for key in self.post_out:
            metrics[key] += self.post_out[key]
        return metrics

    @staticmethod
    def display(metrics, num_preds=30):
        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        key_points = metrics["key_points_loss"] / (metrics["num_key_points"] + 1e-10)
        loss = cls + reg + key_points

        preds = np.concatenate(metrics["preds"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = np.concatenate(metrics["has_preds"], 0)
        ade1, fde1, ade, fde, min_ids = pred_metrics(preds, gt_preds, has_preds, num_preds)
        out = {
            "loss": loss,
            "cls": cls,
            "reg": reg,
            "key_points": key_points,
            "ade1": ade1,
            "fde1": fde1,
            "ade": ade,
            "fde": fde
        }
        return out


def pred_metrics(preds, gt_preds, has_preds, num_preds):
    assert has_preds.all()
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)

    # err: n * 6 * 30
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))

    ade1 = err[:, 0, :num_preds].mean()
    fde1 = err[:, 0, num_preds - 1].mean()

    min_ids = err[:, :, -1].argmin(1)
    row_ids = np.arange(len(min_ids)).astype(np.int64)
    err = err[row_ids, min_ids]
    ade = err[..., :num_preds].mean()
    fde = err[..., num_preds - 1].mean()
    return ade1, fde1, ade, fde, min_ids


def visualization(out, data, num, save, show):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("equal")

    orig = data['orig'][0].detach().cpu().numpy()
    rot = data["rot"][0].detach().cpu().numpy()
    x_min = np.min(orig[0]) - 100
    x_max = np.max(orig[0]) + 100
    y_min = np.min(orig[1]) - 100
    y_max = np.max(orig[1]) + 100
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # map
    ctrs = data["graph"][0]["ctrs"].detach().cpu().numpy()
    ctrs[:, :2] = ctrs[:, :2].dot(rot.T) + orig
    # ax.scatter(ctrs[:, 0], ctrs[:, 1], color="b", s=2, alpha=0.5)

    feats = data["graph"][0]["feats"].detach().cpu().numpy()
    feats[:, :2] = feats[:, :2].dot(rot.T)
    for j in range(feats.shape[0]):
        vec = feats[j]
        pt0 = ctrs[j] - vec / 2
        pt1 = ctrs[j] + vec / 2
        ax.arrow(pt0[0], pt0[1], (pt1 - pt0)[0], (pt1 - pt0)[1], edgecolor=None, color="grey", alpha=0.5)

    # trajectories
    gt_preds = data["trajs_fut"][0][0].detach().cpu().numpy().dot(rot.T) + orig
    pad_fut = data["pad_fut"][0][0].detach().cpu().numpy()
    ax.plot(gt_preds[:-1, 0], gt_preds[:-1, 1], marker=".", alpha=1, color="g", zorder=20)
    ax.arrow(gt_preds[-2, 0], gt_preds[-2, 1], (gt_preds[-1] - gt_preds[-2])[0], (gt_preds[-1] - gt_preds[-2])[1], 
             color="g", alpha=1, width=1)

    trajs_obs = data["trajs_obs"][0][0].detach().cpu().numpy().dot(rot.T) + orig
    pad_obs = data["pad_obs"][0][0].detach().cpu().numpy()
    ax.plot(trajs_obs[:, 0], trajs_obs[:, 1], marker=".", alpha=1, color="r", zorder=20)

    preds = out['reg'][0][0].detach().cpu().numpy().dot(rot.T) + orig
    key_points = out["key_points"][0][0].detach().cpu().numpy().dot(rot.T) + orig
    probabilities = out['cls'][0][0]
    probabilities = probabilities.detach().cpu().numpy() * 100
    for i in range(preds.shape[0]):
        ax.scatter(preds[i, -1, 0], preds[i, -1, 1], marker="X", c="orange", s=10, zorder=20, alpha=1)
        ax.text(preds[i, -1, 0], preds[i, -1, 1], str("%d" % probabilities[i]), fontsize=8, alpha=0.5, 
                horizontalalignment="center", verticalalignment="bottom")

    ade1, fde1, ade, fde, idx = pred_metrics(np.expand_dims(preds, 0), np.expand_dims(gt_preds, 0),
                                                 pad_fut, config["num_preds"])
    ax.text(x_max, y_min, "ade1: {:.3f}\nfde1: {:.3f}\nade6: {:.3f}\nfde6: {:.3f}".format(ade1, fde1, ade, fde),
            fontsize=15, horizontalalignment="right", verticalalignment="bottom")
    ax.scatter(preds[idx, -1, 0], preds[idx, -1, 1], marker="X", c="b", s=10, zorder=20, alpha=1)

    if save:
        plt.savefig(config["images"] + str("%d_%.3f" % (num, fde)) + '.png', dpi=250)
    if show:
        plt.show()


def visualization_for_all_agents(out, data, num, save, show):
    fig, ax = plt.subplots(figsize=(10, 10))
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
    ctrs = data["graph"][0]["ctrs"].detach().cpu().numpy().dot(rot.T) + orig
    ax.scatter(ctrs[:, 0], ctrs[:, 1], color="b", s=2, alpha=0.5)

    feats = data["graph"][0]["feats"].detach().cpu().numpy().dot(rot.T)
    for j in range(feats.shape[0]):
        vec = feats[j]
        pt0 = ctrs[j] - vec / 2
        pt1 = ctrs[j] + vec / 2
        ax.arrow(pt0[0], pt0[1], (pt1 - pt0)[0], (pt1 - pt0)[1], edgecolor=None, color="deepskyblue", alpha=0.3)

    # trajectories
    trajs_obs = data["trajs_obs"][0].detach().cpu().numpy().dot(rot.T) + orig
    pad_obs = data["pad_obs"][0].detach().cpu().numpy()
    mask = pad_obs.astype(np.bool)
    for i in range(trajs_obs.shape[0]):
        ax.plot(trajs_obs[i, :, 0], trajs_obs[i, :, 1], marker=".", alpha=0.5, color=colors[i % 18], zorder=20)
        ax.scatter(trajs_obs[i, mask[i], 0], trajs_obs[i, mask[i], 1], marker="o",
                   c=np.arange(len(np.nonzero(mask[i])[0])), cmap=color_map[i % 18])

    # prediction
    preds = out['reg'][0][:, 0].detach().cpu().numpy().dot(rot.T) + orig
    key_points = out["key_points"][0][:, 0].detach().cpu().numpy().dot(rot.T) + orig
    for i in range(preds.shape[0]):
        ax.scatter(preds[i, :, 0], preds[i, :, 1], marker=".", c=colors[i % 18])
        ax.scatter(key_points[i, -1, 0], key_points[i, -1, 1], marker="X", c=colors[i % 18])

    gt_preds = data['trajs_fut'][0].detach().cpu().numpy().dot(rot.T) + orig
    fde = np.sqrt(np.sum((preds[:, -1] - gt_preds[:, -1]) ** 2, axis=-1))[0]

    if save:
        plt.savefig(config["images"] + str("%d_all_%.3f" % (num, fde)) + '.png', dpi=250)
    if show:
        plt.show()


def create_dirs():
    if not os.path.exists(config["save_dir"]):
        os.makedirs(config["save_dir"])
        print("mkdir " + config["save_dir"])
    if not os.path.exists(config["images"]):
        os.makedirs(config["images"])
        print("mkdir " + config["images"])
    if not os.path.exists(config["competition_files"]):
        os.makedirs(config["competition_files"])
        print("mkdir " + config["competition_files"])
    f = open(config["result"], 'a')
    f.write("---------------------------hyper parameters---------------------------\n")
    for k, v in config.items():
        f.write(k + str(":") + str(v) + '\n')
    f.write("----------------------------------------------------------------------\n")
    f.close()


def save_log(epoch, out, divided=False):
    f = open(config["result"], 'a')
    f.write(str('%.3f' % epoch) + '\t')
    for k, value in out.items():
        f.write(str('%.3f' % value) + '\t')
    f.write("\n")
    if divided:
        f.write("----------------------------------------------------------------------\n")
    f.close()
