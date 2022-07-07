import os
import argparse
import time
import copy

import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed
from torch.utils.data import Dataset


class FlattenDataset(Dataset):
    def __init__(self, src, dst):
        super(FlattenDataset, self).__init__()
        self.src = src
        self.dst = dst
        self.file_list = os.listdir(self.src)
        self.num_obs = 20
        self.count = 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        file = self.file_list[item]
        df = pd.read_pickle(os.path.join(self.src, file))
        # "SEQ_ID", "ORIG", "ROT", "TIMESTAMP", "TRAJS", "PAD_FLAGS", "GRAPH"
        df_dict = {}
        for key in list(df.keys()):
            df_dict[key] = df[key].values[0]

        seq_id = df_dict["SEQ_ID"]
        ts = df_dict["TIMESTAMP"]
        pred_trajs = df_dict["PRED_TRAJS"]
        has_preds = df_dict["HAS_PREDS"]

        trajs = df_dict["TRAJS"]
        pad_flags = df_dict["PAD_FLAGS"]
        graph = df_dict["GRAPH"]

        # chose new vehicle coordinate
        ids = np.nonzero(has_preds)[0]
        self.count += len(ids)
        for pred_idx, has_preds_idx in enumerate(ids):
            # move the chosen agent to the first
            row_ids = np.arange(len(has_preds))
            row_ids[0], row_ids[has_preds_idx] = row_ids[has_preds_idx], row_ids[0]
            p_trajs = trajs[row_ids]
            p_pad_flags = pad_flags[row_ids]
            pred_traj = pred_trajs[pred_idx: pred_idx + 1]
            # transform from agent coordinate to other vehicle coordinate
            p_orig = p_trajs[0, self.num_obs - 1]
            vec = p_orig - p_trajs[0, 0]
            theta = np.arctan2(vec[1], vec[0])
            p_rot = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
            p_trajs = (np.asarray(p_trajs) - p_orig).dot(p_rot)
            pred_traj = (np.asarray(pred_traj) - p_orig).dot(p_rot)
            p_graph = copy.deepcopy(graph)
            p_graph["ctrs"] = (np.asarray(p_graph["ctrs"]) - p_orig).dot(p_rot)
            p_graph["feats"] = (np.asarray(p_graph["feats"])).dot(p_rot)
            # transform from vehicle coordinate to world coordinate
            p_orig = np.matmul(p_orig, df_dict["ROT"]) + df_dict["ORIG"]
            p_rot = np.matmul(p_rot, df_dict["ROT"])

            data = {
                "seq_id": seq_id + "_{}".format(pred_idx),
                "orig": p_orig,
                "rot": p_rot,
                "ts": np.diff(ts, prepend=ts[0])[:self.num_obs],
                "trajs_obs": p_trajs[:, :self.num_obs],
                "pad_obs": p_pad_flags[:, :self.num_obs],
                "trajs_fut": p_trajs[:, self.num_obs:],
                "pad_fut": p_pad_flags[:, self.num_obs:],
                "graph": p_graph,
                "pred_trajs": pred_traj
            }

            save_dir = os.path.join(self.dst, data["seq_id"] + "_argo" + ".pkl")
            keys = [_ for _ in data.keys()]
            values = [[data[key] for key in keys]]
            df = pd.DataFrame(data=values, columns=keys)
            df.to_pickle(save_dir)

        return len(ids)


def save_flattened_data(dataset, start_idx, batch_size):
    loop = tqdm(range(batch_size), total=batch_size, leave=True)
    for i in loop:
        if i + start_idx < len(dataset):
            count = dataset[i + start_idx]
            loop.set_postfix_str(f"saved {dataset.count} files")


def main(args):
    start = time.time()
    src = args.src
    dst = args.dst
    dataset = FlattenDataset(src, dst)
    num_files = len(dataset)
    print("num of files: ", num_files)

    n_proc = multiprocessing.cpu_count() - 2 if not args.debug else 1
    batch_size = np.max([int(np.ceil(num_files / float(n_proc))), 1])
    print('n_proc: {}, batch_size: {}'.format(n_proc, batch_size))

    Parallel(n_jobs=n_proc)(delayed(save_flattened_data)(dataset, i, batch_size)
                            for i in range(0, num_files, batch_size))
    print("Preprocess for {} set completed in {} minutes".format(args.src, (time.time() - start) / 60.0))


def get_args():
    parser = argparse.ArgumentParser(description='Data Flattening')
    parser.add_argument('--src', default='./data/features/baseline/', type=str, help='the path of saved features')
    parser.add_argument('--dst', default='./data/features/flattened_baseline',
                        type=str, help='the path of flattened features')
    parser.add_argument('--debug', action="store_true", default=False, help='whether to debug')
    args = parser.parse_args()
    if not os.path.exists(args.dst):
        os.mkdir(args.dst)
    return args


if __name__ == "__main__":
    main(get_args())
