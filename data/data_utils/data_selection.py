import os
import argparse
import shutil
import time
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed
from torch.utils.data import Dataset


class SelectionDataset(Dataset):
    def __init__(self, path):
        super(SelectionDataset, self).__init__()
        self.obs_len = 20
        self.num_preds = 30

        self.path = path
        file_list = os.listdir(path)
        file_list.sort(key=lambda x: int(x.split(".")[0].split("_")[-3]) +
                                     len(file_list) * int(x.split(".")[0].split("_")[-2]))
        self.file_list = file_list
        self.selected_files = {
            "2": [],
            "4": [],
            "6": [],
            "8": [],
            "10": []
        }

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        file = self.file_list[item]
        df = pd.read_pickle(os.path.join(self.path, file))
        # "SEQ_ID", "ORIG", "ROT", "TIMESTAMP", "TRAJS", "PAD_FLAGS", "GRAPH"
        df_dict = {}
        for key in ["SEQ_ID", "TRAJS", "PAD_FLAGS"]:
            df_dict[key] = df[key].values[0]

        seq_id = df_dict["SEQ_ID"]
        trajs = df_dict["TRAJS"]
        pad_flags = df_dict["PAD_FLAGS"]

        trajs_obs = trajs[:, :self.obs_len]
        trajs_fut = trajs[:, self.obs_len:]
        pad_obs = pad_flags[:, :self.obs_len]
        pad_fut = pad_flags[:, self.obs_len:]

        if not np.all(np.argmax(pad_obs, axis=1) >= np.argmin(pad_obs, axis=1)) or not np.all(pad_obs[:, -2]):
            return sum([len(v) for k, v in self.selected_files.items()]), seq_id

        feats = np.zeros_like(trajs_obs)
        feats[:, 1:] = trajs_obs[:, 1:] - trajs_obs[:, :-1]
        feats = np.concatenate([feats, np.expand_dims(pad_obs, 2)], axis=-1)
        feats[:, :, :2] *= feats[:, :, -1:]
        feats[:, 1:, :2] *= feats[:, :-1, -1:]
        speed = feats.sum(1) / (feats[:, :, -1:].sum(1) - 1 + 1e-10)

        trajs_pred = np.repeat(trajs_obs[:, -1:], self.num_preds, axis=1)
        for i in range(self.num_preds):
            trajs_pred[:, i] = trajs_pred[:, i] + speed[:, :2] * (i + 1)

        last = pad_fut.astype(np.float_) + 0.1 * np.arange(self.num_preds).astype(np.float_) / self.num_preds
        max_last, last_ids = np.max(last, axis=-1), np.argmax(last, axis=-1)
        mask = max_last > 1.0

        trajs_pred = trajs_pred[mask]
        trajs_fut = trajs_fut[mask]
        last_ids = last_ids[mask]

        row_ids = np.arange(len(last_ids)).astype(np.int16)
        dist = np.sqrt(((trajs_pred[row_ids, last_ids] - trajs_fut[row_ids, last_ids]) ** 2).sum(1))

        if dist.mean() <= 2.0:
            self.selected_files["2"].append(seq_id)
        elif 2.0 < dist.mean() <= 4.0:
            self.selected_files["4"].append(seq_id)
        elif 4.0 < dist.mean() <= 6.0:
            self.selected_files["6"].append(seq_id)
        elif 6.0 < dist.mean() <= 8.0:
            self.selected_files["8"].append(seq_id)
        elif 8.0 < dist.mean() <= 10.0:
            self.selected_files["10"].append(seq_id)

        return sum([len(v) for k, v in self.selected_files.items()]), seq_id


def move_files(files, path, dst):
    print(sum([len(v) for k, v in files.items()]))
    for k, v in files.items():
        saved_path = os.path.join(dst, str(k))
        if not os.path.exists(saved_path):
            os.mkdir(saved_path)
        for file in v:
            if os.path.exists(path + file + "_argo.pkl"):
                shutil.copyfile(path + file + "_argo.pkl", os.path.join(saved_path, file + "_argo.pkl"))
            else:
                print("file: {} not found".format(path + file + "_argo.pkl"))


def search_for_tough_case(dataset, start_idx, batch_size, path, dst):
    loop = tqdm(range(batch_size), total=batch_size, leave=True)
    for i in loop:
        if i + start_idx < len(dataset):
            count, seq_id = dataset[i + start_idx]
            loop.set_description(f"current file: {seq_id}")
            loop.set_postfix_str(f"count: {count}")
    move_files(dataset.selected_files, path, dst)


def main(args):
    start = time.time()

    path = args.src
    dst = args.dst
    dataset = SelectionDataset(path)
    num_files = len(dataset)
    print("num of files: ", num_files)

    n_proc = multiprocessing.cpu_count() - 2 if not args.debug else 1
    batch_size = np.max([int(np.ceil(num_files / float(n_proc))), 1])
    print('n_proc: {}, batch_size: {}'.format(n_proc, batch_size))

    Parallel(n_jobs=n_proc)(delayed(search_for_tough_case)(dataset, i, batch_size, path, dst)
                            for i in range(0, num_files, batch_size))
    print("Preprocess for {} set completed in {} minutes".format(args.src, (time.time() - start) / 60.0))


def get_args():
    parser = argparse.ArgumentParser(description='Data Cleaning')
    parser.add_argument('--src', default='./data/features/baseline/', type=str, help='the path of saved features')
    parser.add_argument('--dst', default='./data/features/selected_baseline/',
                        type=str, help='the path of selected samples')
    parser.add_argument('--debug', action="store_true", default=False, help='whether to debug')
    args = parser.parse_args()
    if not os.path.exists(args.dst):
        os.mkdir(args.dst)
    return args


if __name__ == "__main__":
    main(get_args())
