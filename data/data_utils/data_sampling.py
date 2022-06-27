import os
import shutil
import argparse
from tqdm import trange


def get_args():
    parser = argparse.ArgumentParser(description="Data Sampling")
    parser.add_argument("--src", default="data/features/train", type=str, help="the path of source files")
    parser.add_argument("--dst", default="data/features/sampled_train", type=str, help="the path of sampled files")
    parser.add_argument("--interval", default=20, type=int, help="sampling interval")
    return parser.parse_args()


class SamplingDataset(object):
    def __init__(self, src, dst, interval=10):
        super(SamplingDataset, self).__init__()
        self.src = src
        self.dst = dst
        self.interval = interval

        self.file_list = os.listdir(self.src)
        self.bag_dict = self.get_bag_dict()
        self.sorted_file_list = self.get_sorted_file_list()

    def get_bag_dict(self):
        bag_dict = {}
        count = 0
        for file in self.file_list:
            key = self.get_bag_name_and_frame(file)[0]
            if key not in bag_dict.keys():
                bag_dict[key] = count
                count += 1
        return bag_dict

    @staticmethod
    def get_bag_name_and_frame(x):
        index = sum([len(x.split("_")[i]) for i in range(-3, 0)]) + 3
        bag_name = x[:-index]
        frame = int(x.split("_")[-3])
        return bag_name, frame

    def get_sorted_file_list(self):
        sorted_file_list = sorted(self.file_list, key=lambda x: self.bag_dict[self.get_bag_name_and_frame(x)[0]] *
                                                                len(self.file_list) + self.get_bag_name_and_frame(x)[1])
        return sorted_file_list

    def start(self):
        for i in trange(0, len(self.sorted_file_list), self.interval):
            src = os.path.join(self.src, self.sorted_file_list[i])
            dst = os.path.join(self.dst, self.sorted_file_list[i])
            shutil.copyfile(src, dst)


if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.dst):
        os.mkdir(args.dst)
    sd = SamplingDataset(args.src, args.dst, args.interval)
    sd.start()
