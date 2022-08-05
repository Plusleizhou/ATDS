import numpy as np
import torch
from load_config import config
import matplotlib.pyplot as plt


class LeadPostProcess(object):
    def __init__(self, out=None, data=None):
        # agent trajectory is in the first pos
        self.post_out = dict()
        if out is not None:
            self.post_out['preds'] = [x[mask == 2].detach().cpu().numpy()
                                      for x, mask in zip(out['reg'], data["ego_lane_presence"])]
        if data is not None:
            self.post_out["gt_preds"] = [x[mask == 2].numpy() for x, mask in
                                         zip(data["trajs_fut"], data["ego_lane_presence"])]
            self.post_out["has_preds"] = [x[mask == 2].numpy() for x, mask in
                                          zip(data["pad_fut"], data["ego_lane_presence"])]
            self.post_out["pred_trajs"] = [x[ego_lane_presence[mask == 1] == 2].numpy() for x, ego_lane_presence, mask
                                           in zip(data["pred_trajs"], data["ego_lane_presence"], data["has_preds"])]

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
    def display(metrics, num_preds=30, multi_range=False):
        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        key_points = metrics["key_points_loss"] / (metrics["num_key_points"] + 1e-10)
        loss = cls + reg + key_points

        preds = np.concatenate(metrics["preds"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = np.concatenate(metrics["has_preds"], 0)
        pred_trajs = np.concatenate(metrics["pred_trajs"], 0)
        base_ade1, base_fde1, ade1, fde1, ade, fde, min_ids = pred_metrics(preds, pred_trajs, gt_preds,
                                                                           has_preds, num_preds)
        out = {
            "loss": loss,
            "cls": cls,
            "reg": reg,
            "key_points": key_points,
            "base_ade1": base_ade1,
            "base_fde1": base_fde1,
            "ade1": ade1,
            "fde1": fde1,
            "ade": ade,
            "fde": fde
        }
        return out


def pred_metrics(preds, pred_trajs, gt_preds, has_preds, num_preds):
    has_preds = (has_preds == 1)[:, :num_preds]
    has_final_preds = has_preds[:, num_preds - 1]  # which has final point

    # baseline
    base_ade1 = np.sqrt((np.sum((gt_preds[:, :num_preds][has_preds] -
                                pred_trajs[:, :num_preds][has_preds]) ** 2, axis=-1))).mean()
    base_fde1 = np.sqrt((np.sum((gt_preds[:, num_preds - 1][has_final_preds] -
                                pred_trajs[:, num_preds - 1][has_final_preds]) ** 2, axis=-1))).mean()

    # model
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)

    ade1 = np.sqrt((np.sum((gt_preds[:, :num_preds][has_preds] -
                            preds[:, 0, :num_preds][has_preds]) ** 2, axis=-1))).mean()
    fde1 = np.sqrt((np.sum((gt_preds[:, num_preds - 1][has_final_preds] -
                            preds[:, 0, num_preds - 1][has_final_preds]) ** 2, axis=-1))).mean()

    gt_preds = gt_preds[has_final_preds]
    preds = preds[has_final_preds]
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))

    min_ids = err[:, :, num_preds - 1].argmin(1)
    row_ids = np.arange(len(min_ids)).astype(np.int64)
    err = err[row_ids, min_ids]
    ade = err[..., :num_preds].mean()
    fde = err[..., num_preds - 1].mean()
    return base_ade1, base_fde1, ade1, fde1, ade, fde, min_ids


def visualization(out, data, num, save, show):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(data["seq_id"][0], fontweight="bold")
    ax.axis("equal")

    data["trajs_obs"] = [x[:, :, :2] for x in data["trajs_obs"]]

    orig = data['orig'][0].detach().cpu().numpy()
    rot = data["rot"][0].detach().cpu().numpy()

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
    ego_gt_preds = data["trajs_fut"][0][0].detach().cpu().numpy().dot(rot.T) + orig
    ax.plot(ego_gt_preds[:-1, 0], ego_gt_preds[:-1, 1], marker=".", alpha=1, color="g", zorder=20)
    ax.arrow(ego_gt_preds[-2, 0], ego_gt_preds[-2, 1], (ego_gt_preds[-1] - ego_gt_preds[-2])[0],
             (ego_gt_preds[-1] - ego_gt_preds[-2])[1], color="g", alpha=1, width=1)

    ego_trajs_obs = data["trajs_obs"][0][0].detach().cpu().numpy().dot(rot.T) + orig
    ax.plot(ego_trajs_obs[:, 0], ego_trajs_obs[:, 1], marker=".", alpha=1, color="r", zorder=20)

    # surrounding agents
    has_preds = data["has_preds"][0].detach().cpu().numpy().astype(np.bool)
    ego_lane_presence = data["ego_lane_presence"][0].detach().cpu().numpy()
    pred_trajs = data["pred_trajs"][0][ego_lane_presence[has_preds] == 2].detach().cpu().numpy().dot(rot.T) + orig

    gt_preds = data["trajs_fut"][0][ego_lane_presence == 2].detach().cpu().numpy().dot(rot.T) + orig
    pad_fut = data["pad_fut"][0][ego_lane_presence == 2].detach().cpu().numpy() == 1
    trajs_obs = data["trajs_obs"][0][ego_lane_presence == 2].detach().cpu().numpy().dot(rot.T) + orig
    pad_obs = data["pad_obs"][0][ego_lane_presence == 2].detach().cpu().numpy().astype(np.bool)
    preds = out['reg'][0][ego_lane_presence == 2].detach().cpu().numpy().dot(rot.T) + orig
    probabilities = out['cls'][0][ego_lane_presence == 2].detach().cpu().numpy() * 100

    for i in range(gt_preds.shape[0]):
        gt_pred = gt_preds[i][pad_fut[i]]
        if gt_pred.shape[0] == 0:
            continue
        traj_obs = trajs_obs[i][pad_obs[i]]
        pred_traj = pred_trajs[i, :pad_fut[i].shape[0]][pad_fut[i]]
        ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1], marker="X", c=colors[i % 18], s=30, zorder=20, alpha=1)
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

    all_points = np.concatenate([ego_trajs_obs, ego_gt_preds, gt_preds[0], trajs_obs[0]], axis=0)
    x_min = np.min(all_points[..., 0]) - 20
    x_max = np.max(all_points[..., 0]) + 20
    y_min = np.min(all_points[..., 1]) - 20
    y_max = np.max(all_points[..., 1]) + 20
    center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
    width = np.max([x_max - x_min, y_max - y_min])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    base_ade1, base_fde1, ade1, fde1, ade, fde, min_ids = pred_metrics(
        preds, pred_trajs, gt_preds, pad_fut, config["num_preds"])
    ax.text(center[0] + width / 2, center[1] - width / 2,
            "b_ade1:{:.3f}\nb_fde1:{:.3f}\nade1: {:.3f}\nfde1: {:.3f}\nade6: {:.3f}\nfde6: {:.3f}".
            format(base_ade1, base_fde1, ade1, fde1, ade, fde),
            fontsize=15, horizontalalignment="right", verticalalignment="bottom")

    preds = preds[pad_fut[:, config["num_preds"] - 1]]
    for i, idx in enumerate(min_ids):
        ax.scatter(preds[i, idx, -1, 0], preds[i, idx, -1, 1], marker="X", c="b", s=30, zorder=20, alpha=1)
        ax.text(preds[i, idx, -1, 0], preds[i, idx, -1, 1], str("%d" % probabilities[i, idx]), fontsize=8, alpha=1,
                horizontalalignment="center", verticalalignment="bottom")

    if save:
        plt.savefig(config["images"] + str("%d_lead_%.3f" % (num, fde)) + '.png', dpi=250)
    if show:
        plt.show()
    plt.close(fig)
