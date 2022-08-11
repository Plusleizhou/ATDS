import torch
import torch.nn as nn
from fractions import gcd
from model.Layers import Linear, index_add_naive


class MapEncoder(nn.Module):
    def __init__(self, config):
        super(MapEncoder, self).__init__()
        self.config = config
        n_map = config["n_map"]
        ng = 1

        in_dim = 2

        self.input = nn.Sequential(
            nn.Linear(in_dim, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, ng=ng, act=False),
        )
        self.seg = nn.Sequential(
            nn.Linear(in_dim, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, ng=ng, act=False),
        )

        self.meta = Linear(n_map + 4, n_map, ng=ng)

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(ng, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(n_map, n_map, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, nodes, indexes):
        feat = self.input(nodes[:, :2])
        feat += self.seg(nodes[:, 2:4])
        feat = self.relu(feat)

        feat = self.meta(torch.cat((feat, nodes[:, 4:]), 1))

        res = feat
        for i in range(4):
            temp = self.fuse["ctr"][i](feat)
            temp.index_add_(0, indexes[:, 0], self.fuse["pre0"][i](feat[indexes[:, 1]]))
            temp.index_add_(0, indexes[:, 2], self.fuse["pre1"][i](feat[indexes[:, 3]]))
            temp.index_add_(0, indexes[:, 4], self.fuse["pre2"][i](feat[indexes[:, 5]]))
            temp.index_add_(0, indexes[:, 6], self.fuse["pre3"][i](feat[indexes[:, 7]]))
            temp.index_add_(0, indexes[:, 8], self.fuse["pre4"][i](feat[indexes[:, 9]]))
            temp.index_add_(0, indexes[:, 10], self.fuse["pre5"][i](feat[indexes[:, 11]]))

            temp.index_add_(0, indexes[:, 12], self.fuse["right"][i](feat[indexes[:, 13]]))

            temp.index_add_(0, indexes[:, 14], self.fuse["suc0"][i](feat[indexes[:, 15]]))
            temp.index_add_(0, indexes[:, 16], self.fuse["suc1"][i](feat[indexes[:, 17]]))
            temp.index_add_(0, indexes[:, 18], self.fuse["suc2"][i](feat[indexes[:, 19]]))
            temp.index_add_(0, indexes[:, 20], self.fuse["suc3"][i](feat[indexes[:, 21]]))
            temp.index_add_(0, indexes[:, 22], self.fuse["suc4"][i](feat[indexes[:, 23]]))
            temp.index_add_(0, indexes[:, 24], self.fuse["suc5"][i](feat[indexes[:, 25]]))

            temp.index_add_(0, indexes[:, 26], self.fuse["left"][i](feat[indexes[:, 27]]))

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat, nodes[:, :2]
