import os
import argparse
import shutil
import time
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
from torch.utils.data import Dataset


def get_val_dict(path="/home/plusai/Downloads/val_files.txt"):
    with open(path, "rt") as f:
        lines = f.readlines()
    lines = [os.path.basename(line.strip()).replace(".db", "").replace("agent_", "").split(".")[0] for line in lines]
    val_dict = defaultdict(list)
    for line in lines:
        data = line.split("_")
        ts = data[-2]
        obs_id = data[-3]
        bag_name = line[:-(sum([len(_) for _ in data[-3:]]) + 3)]
        val_dict[bag_name + "_" + ts + "_-1"].append(int(obs_id))
    return val_dict


class SelectionDataset(Dataset):
    def __init__(self, path, val_dict):
        super(SelectionDataset, self).__init__()
        self.obs_len = 20
        self.num_preds = 30

        self.path = path
        file_list = os.listdir(path)
        file_list.sort(key=lambda x: int(x.split(".")[0].split("_")[-3]) +
                                     len(file_list) * int(x.split(".")[0].split("_")[-2]))
        self.file_list = file_list
        self.val_dict = val_dict
        self.selected_files = []

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        file = self.file_list[item]
        df = pd.read_pickle(os.path.join(self.path, file))
        # "SEQ_ID", "ORIG", "ROT", "TIMESTAMP", "TRAJS", "PAD_FLAGS", "GRAPH"
        df_dict = {}
        for key in ["SEQ_ID", "AGENT_IDS"]:
            df_dict[key] = df[key].values[0]

        seq_id = df_dict["SEQ_ID"]

        if seq_id in self.val_dict:
            self.selected_files.append(seq_id)

        return len(self.selected_files), seq_id


def move_files(files, path, dst):
    print(len(files))
    for file in files:
        if os.path.exists(path + file + "_argo.pkl"):
            shutil.copyfile(path + file + "_argo.pkl", os.path.join(dst, file + "_argo.pkl"))
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

    val_dict = get_val_dict(args.txt)
    print("num of val files: ", len(val_dict.keys()))

    path = args.src
    dst = args.dst
    dataset = SelectionDataset(path, val_dict)
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
    parser.add_argument('--src', default='./data/features/id_val/',
                        type=str, help='the path of saved features')
    parser.add_argument('--dst', default='./data/features/selected_val/',
                        type=str, help='the path of selected samples')
    parser.add_argument('--txt', default='/home/plusai/Downloads/val_files.txt',
                        type=str, help='the path of txt file of selected samples')
    parser.add_argument('--debug', action="store_true", default=False, help='whether to debug')
    args = parser.parse_args()
    if not os.path.exists(args.dst):
        os.mkdir(args.dst)
    return args


if __name__ == "__main__":
    main(get_args())

