import os
import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import trange
from load_config import config
from torch.utils.data import Dataset


class ProcessedDataset(Dataset):
    def __init__(self, path, mode):
        super(ProcessedDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.train = False
        if self.mode == "train":
            self.train = True
        file_list = os.listdir(path)
        file_list.sort(key=lambda x: int(x.split(".")[0].split("_")[-3]) +
                                                       len(file_list) * int(x.split(".")[0].split("_")[-2]))
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        file = self.file_list[item]
        df = pd.read_pickle(os.path.join(self.path, file))
        # "SEQ_ID", "ORIG", "ROT", "TIMESTAMP", "TRAJS", "PAD_FLAGS", "GRAPH"
        df_dict = {}
        for key in list(df.keys()):
            df_dict[key] = df[key].values[0]

        seq_id = df_dict["SEQ_ID"]
        orig = df_dict["ORIG"]
        rot = df_dict["ROT"]
        ts = df_dict["TIMESTAMP"]

        trajs = df_dict["TRAJS"]
        pad_flags = df_dict["PAD_FLAGS"]
        graph = df_dict["GRAPH"]

        update_mask = np.ones(trajs.shape[0]).astype(np.bool_)

        data = {
            "seq_id": seq_id,
            "orig": orig,
            "rot": rot,
            "ts": np.diff(ts, prepend=ts[0])[:config["num_obs"]],
            "trajs_obs": trajs[:, :config["num_obs"]],
            "pad_obs": pad_flags[:, :config["num_obs"]],
            "trajs_fut": trajs[:, config["num_obs"]:, :2],
            "pad_fut": pad_flags[:, config["num_obs"]:],
            "graph": graph,
            "update_mask": update_mask
        }

        if self.train is True:
            data = DataAug.simple_aug(data)
            # data = DataAug.transform(data)
        return data


class SeqProcessedDataset(Dataset):
    def __init__(self, path, mode):
        super(SeqProcessedDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.train = False
        if self.mode == "train":
            self.train = True
        file_list = os.listdir(path)
        self.file_list = sorted(file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        file = self.file_list[item]
        df = pd.read_pickle(os.path.join(self.path, file))
        data = {}
        for key in list(df.keys()):
            data[key] = df[key].values[0]
        data["update_mask"] = np.ones(data["trajs_obs"].shape[0]).astype(np.bool_)
        return data


class BaseProcessedDataset(Dataset):
    def __init__(self, path, mode="val"):
        super(BaseProcessedDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.train = False
        if self.mode == "train":
            self.train = True
        file_list = os.listdir(path)
        file_list.sort(key=lambda x: int(x.split(".")[0].split("_")[-3]) +
                                                       len(file_list) * int(x.split(".")[0].split("_")[-2]))
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        file = self.file_list[item]
        df = pd.read_pickle(os.path.join(self.path, file))
        # "SEQ_ID", "ORIG", "ROT", "TIMESTAMP", "TRAJS", "PAD_FLAGS", "GRAPH"
        df_dict = {}
        for key in list(df.keys()):
            df_dict[key] = df[key].values[0]

        seq_id = df_dict["SEQ_ID"]
        orig = df_dict["ORIG"]
        rot = df_dict["ROT"]
        ts = df_dict["TIMESTAMP"]

        trajs = df_dict["TRAJS"]
        trajs_obs = trajs[:, :config["num_obs"]]
        trajs_fut = trajs[:, config["num_obs"]:]

        pad_flags = df_dict["PAD_FLAGS"]
        pad_obs = pad_flags[:, :config["num_obs"]]
        pad_fut = pad_flags[:, config["num_obs"]:]

        has_preds = df_dict["HAS_PREDS"]
        pred_trajs = df_dict["PRED_TRAJS"].astype(np.float32)

        graph = df_dict["GRAPH"]

        update_mask = np.ones(trajs.shape[0]).astype(np.bool_)

        data = {
            "seq_id": seq_id,
            "orig": orig,
            "rot": rot,
            "ts": np.diff(ts, prepend=ts[0])[:config["num_obs"]],
            "trajs_obs": trajs_obs,
            "pad_obs": pad_obs,
            "trajs_fut": trajs_fut[:, :, :2],
            "pad_fut": pad_fut,
            "graph": graph,
            "has_preds": has_preds,
            "pred_trajs": pred_trajs,
            "update_mask": update_mask
        }

        if self.train is True:
            data = DataAug.simple_aug(data)
            # data = DataAug.transform(data)
        return data


class LeadProcessedDataset(Dataset):
    def __init__(self, path, mode="val"):
        super(LeadProcessedDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.train = False
        if self.mode == "train":
            self.train = True
        file_list = os.listdir(path)
        file_list.sort(key=lambda x: int(x.split(".")[0].split("_")[-3]) +
                                                       len(file_list) * int(x.split(".")[0].split("_")[-2]))
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        file = self.file_list[item]
        df = pd.read_pickle(os.path.join(self.path, file))
        # "SEQ_ID", "ORIG", "ROT", "TIMESTAMP", "TRAJS", "PAD_FLAGS", "GRAPH"
        df_dict = {}
        for key in list(df.keys()):
            df_dict[key] = df[key].values[0]

        seq_id = df_dict["SEQ_ID"]
        orig = df_dict["ORIG"]
        rot = df_dict["ROT"]
        ts = df_dict["TIMESTAMP"]

        trajs = df_dict["TRAJS"]

        pad_flags = df_dict["PAD_FLAGS"]
        pad_obs = pad_flags[:, :config["num_obs"]]
        pad_fut = pad_flags[:, config["num_obs"]:]

        has_preds = df_dict["HAS_PREDS"]
        pred_trajs = df_dict["PRED_TRAJS"].astype(np.float32)
        ego_lane_presence = df_dict["EGO_LANE_PRESENCE"]

        graph = df_dict["GRAPH"]

        update_mask = np.ones(trajs.shape[0]).astype(np.bool_)

        data = {
            "seq_id": seq_id,
            "orig": orig,
            "rot": rot,
            "ts": np.diff(ts, prepend=ts[0])[:config["num_obs"]],
            "trajs_obs": trajs[:, :config["num_obs"]],
            "pad_obs": pad_obs,
            "trajs_fut": trajs[:, config["num_obs"]:, :2],
            "pad_fut": pad_fut,
            "graph": graph,
            "has_preds": has_preds,
            "pred_trajs": pred_trajs,
            "update_mask": update_mask,
            "ego_lane_presence": ego_lane_presence
        }

        if self.train is True:
            data = DataAug.simple_aug(data)
            # data = DataAug.transform(data)
        return data


class DataAug:
    @classmethod
    def simple_aug(cls, data):
        # speed augmentation
        scaling_ratio = np.random.rand() * (config["scaling_ratio"][1] - config["scaling_ratio"][0]) + \
                        config["scaling_ratio"][0]
        data["trajs_obs"] *= scaling_ratio
        data["graph"]["ctrs"] *= scaling_ratio
        data["graph"]["feats"] *= scaling_ratio
        data["scaling_ratio"] = scaling_ratio
        # past motion dropout
        dropout = np.random.randint(0, config["past_motion_dropout"], size=len(data["trajs_obs"]))
        for i, traj in enumerate(data["trajs_obs"]):
            traj[:dropout[i]] = 0
            data["pad_obs"][i][:dropout[i]] = 0
        # flip
        flip = np.random.rand()
        data["flip"] = flip
        if flip < config["flip"]:
            data["trajs_obs"][:, :, :2] = data["trajs_obs"][:, :, :2][..., ::-1].copy()
            data["graph"]["ctrs"] = data["graph"]["ctrs"][..., ::-1].copy()
            data["graph"]["feats"] = data["graph"]["feats"][..., ::-1].copy()
        return data

    @classmethod
    def transform(cls, data):
        idx = np.random.choice(np.nonzero(data["update_mask"])[0], 1)[0]
        # transform from agent coordinate to other vehicle coordinate
        orig = data["trajs_obs"][idx, config["num_obs"] - 1, :2]
        vec = orig - data["trajs_obs"][idx, 0, :2]
        theta = np.arctan2(vec[1], vec[0])
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        data["trajs_obs"][:, :, :2] = (np.asarray(data["trajs_obs"][:, :, :2]) - orig).dot(rot)
        data["trajs_obs"][:, :, 2:4] = np.asarray(data["trajs_obs"][:, :, 2:4]).dot(rot)
        data["trajs_fut"][:, :, :2] = (np.asarray(data["trajs_fut"][:, :, :2]) - orig).dot(rot)
        data["graph"]["ctrs"] = (np.asarray(data["graph"]["ctrs"]) - orig).dot(rot)
        data["graph"]["feats"] = (np.asarray(data["graph"]["feats"])).dot(rot)
        # transform from vehicle coordinate to world coordinate
        data["orig"] = np.matmul(orig, data["rot"]) + data["orig"]
        data["rot"] = np.matmul(rot, data["rot"])
        return data


def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data


def collate_fn(batch):
    # [{"feats": ...}, {"feats": ...}, ...] ->
    # -> {"feats": [..., ..., ...], "locs": [..., ..., ...], ...}
    batch = from_numpy(batch)
    return_batch = dict()
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch


def from_numpy(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
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


def gpu(data):
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key: gpu(_data) for key, _data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data


if __name__ == "__main__":
    dataset = ProcessedDataset(config["processed_val"], mode="train")
    print(dataset[0])
