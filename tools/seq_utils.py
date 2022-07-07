import numpy as np
import torch
from load_config import config
import matplotlib.pyplot as plt


class SeqPostProcess(object):
    def __init__(self, out=None, data=None):
        # agent trajectory is in the first pos
        self.post_out = dict()
        if out is not None:
            self.post_out['preds'] = [x[0:1].detach().cpu().numpy() for x in out['reg']]
        if data is not None:
            self.post_out["gt_preds"] = [x[0:1].numpy() for x in data["trajs_fut"]]
            self.post_out["pred_trajs"] = [x[0:1].numpy() for x in data["pred_trajs"]]
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
        pred_trajs = np.concatenate(metrics["pred_trajs"], 0)
        base_ade1, base_fde1, ade1, fde1, ade, fde, min_ids = roi_pred_metrics(preds, pred_trajs, gt_preds,
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


def roi_pred_metrics(preds, pred_trajs, gt_preds, has_preds, num_preds):
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
    ax.plot(gt_preds[:-1, 0], gt_preds[:-1, 1], marker=".", alpha=1, color="g", zorder=20, label="Future trajectory")
    ax.arrow(gt_preds[-2, 0], gt_preds[-2, 1], (gt_preds[-1] - gt_preds[-2])[0], (gt_preds[-1] - gt_preds[-2])[1],
             color="g", alpha=1, width=1)

    pred_trajs = data["pred_trajs"][0][0].detach().cpu().numpy().dot(rot.T) + orig
    ax.scatter(pred_trajs[29, 0], pred_trajs[29, 1], marker="X", c="red", s=20, zorder=20, alpha=1,
               label="Predicted goal (baseline)")

    trajs_obs = data["trajs_obs"][0][0].detach().cpu().numpy().dot(rot.T) + orig
    pad_obs = data["pad_obs"][0][0].detach().cpu().numpy()
    ax.plot(trajs_obs[:, 0], trajs_obs[:, 1], marker=".", alpha=1, color="r", zorder=20, label="Historical trajectory")

    preds = out['reg'][0][0].detach().cpu().numpy().dot(rot.T) + orig
    key_points = out["key_points"][0][0].detach().cpu().numpy().dot(rot.T) + orig
    probabilities = out['cls'][0][0]
    probabilities = probabilities.detach().cpu().numpy() * 100
    for i in range(preds.shape[0]):
        ax.scatter(preds[i, -1, 0], preds[i, -1, 1], marker="X", c="orange", s=10, zorder=20, alpha=1)
        ax.text(preds[i, -1, 0], preds[i, -1, 1], str("%d" % probabilities[i]), fontsize=8, alpha=0.5,
                horizontalalignment="center", verticalalignment="bottom")

    base_ade1, base_fde1, ade1, fde1, ade, fde, idx = roi_pred_metrics(np.expand_dims(preds, 0),
                                                                           np.expand_dims(pred_trajs, 0),
                                                                           np.expand_dims(gt_preds, 0),
                                                                           np.expand_dims(pad_fut, 0),
                                                                           config["num_preds"])
    ax.text(x_max, y_min, "b_ade1:{:.3f}\nb_fde1:{:.3f}\nade1: {:.3f}\nfde1: {:.3f}\nade6: {:.3f}\nfde6: {:.3f}".
            format(base_ade1, base_fde1, ade1, fde1, ade, fde),
            fontsize=15, horizontalalignment="right", verticalalignment="bottom")
    if len(idx) > 0:
        ax.scatter(preds[idx, -1, 0], preds[idx, -1, 1], marker="X", c="b", s=20, zorder=20, alpha=1,
                   label="Predicted goal (model)")
        ax.text(preds[idx, -1, 0], preds[idx, -1, 1], str("%d" % probabilities[idx]), fontsize=8, alpha=1,
                horizontalalignment="center", verticalalignment="bottom")

    ax.legend(loc="upper right", shadow=True)

    if save:
        plt.savefig(config["images"] + str("%d_%.3f" % (num, fde)) + '.png', dpi=250)
    if show:
        plt.show()
