import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
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

        if trajs.shape[1] != 50:
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


def main(path):
    dataset = CleaningDataset(path)
    loop = tqdm(range(len(dataset)), total=len(dataset), leave=True)
    for i in loop:
        count, seq_id = dataset[i]
        loop.set_description(f"current file: {seq_id}")
        loop.set_postfix_str(f"count: {count}")
    remove_files(dataset.remove_files, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Cleaning')
    parser.add_argument('--path', default='./data/features/train/', type=str, help='the path of saved features')
    args = parser.parse_args()
    main(args.path)
