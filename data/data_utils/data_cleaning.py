import os
import argparse
import time
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed
from torch.utils.data import Dataset


class CleaningDataset(Dataset):
    def __init__(self, path):
        super(CleaningDataset, self).__init__()
        self.path = path
        file_list = os.listdir(path)
        file_list.sort(key=lambda x: int(x.split(".")[0].split("_")[-3]) +
                                     len(file_list) * int(x.split(".")[0].split("_")[-2]))
        self.file_list = file_list
        self.remove_files = []

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        file = self.file_list[item]
        df = pd.read_pickle(os.path.join(self.path, file))
        # "SEQ_ID", "ORIG", "ROT", "TIMESTAMP", "TRAJS", "PAD_FLAGS", "GRAPH"
        df_dict = {}
        for key in ["SEQ_ID", "TRAJS", "GRAPH"]:
            df_dict[key] = df[key].values[0]

        seq_id = df_dict["SEQ_ID"]
        trajs = df_dict["TRAJS"]
        ctrs = df_dict["GRAPH"]["ctrs"]
        feats = df_dict["GRAPH"]["feats"]

        # Due to the same id of ego and surrounding agents, the length of agent's trajectory will more than 5s.
        # Remove the data which has the abnormal trajectories.
        # Remove the data which have nan in trajectories or graph
        if trajs.shape[1] != 50:
            self.remove_files.append(seq_id)
        elif np.isnan(trajs.max()) or np.isnan(trajs.min()):
            self.remove_files.append(seq_id)
        elif max(np.abs(trajs.max()), np.abs(trajs.min())) > 500:
            self.remove_files.append(seq_id)
        elif len(np.nonzero(np.isnan(ctrs))[0]) > 0 or len(np.nonzero(np.isnan(feats))[0]) > 0:
            self.remove_files.append(seq_id)
        elif max(np.abs(ctrs.max()), np.abs(ctrs.min())) > 1500:
            self.remove_files.append(seq_id)

        return len(self.remove_files), seq_id


def remove_files(files, path):
    print(len(files))
    for file in files:
        if os.path.exists(path + file + "_argo.pkl"):
            os.remove(path + file + "_argo.pkl")
            print("removed {}".format(path + file + "_argo.pkl"))
        else:
            print("file: {} not found".format(path + file + "_argo.pkl"))


def search_for_bad_case(dataset, start_idx, batch_size, path):
    loop = tqdm(range(batch_size), total=batch_size, leave=True)
    for i in loop:
        if i + start_idx < len(dataset):
            count, seq_id = dataset[i + start_idx]
            loop.set_description(f"current file: {seq_id}")
            loop.set_postfix_str(f"count: {count}")
    remove_files(dataset.remove_files, path)


def main(args):
    start = time.time()

    path = args.path
    dataset = CleaningDataset(path)
    num_files = len(dataset)
    print("num of files: ", num_files)

    n_proc = multiprocessing.cpu_count() - 2 if not args.debug else 1
    batch_size = np.max([int(np.ceil(num_files / float(n_proc))), 1])
    print('n_proc: {}, batch_size: {}'.format(n_proc, batch_size))

    Parallel(n_jobs=n_proc)(delayed(search_for_bad_case)(dataset, i, batch_size, path)
                            for i in range(0, num_files, batch_size))
    print("Preprocess for {} set completed in {} minutes".format(args.path, (time.time() - start) / 60.0))


def get_args():
    parser = argparse.ArgumentParser(description='Data Cleaning')
    parser.add_argument('--path', default='./data/features/train/', type=str, help='the path of saved features')
    parser.add_argument('--debug', action="store_true", default=False, help='whether to debug')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(get_args())
