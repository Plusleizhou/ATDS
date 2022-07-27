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

    def forward(self, control, pre, right, suc, turn, intersect, ctrs, feats, left):
        feat = self.input(ctrs)
        feat += self.seg(feats)
        feat = self.relu(feat)

        meta = torch.cat(
            (
                turn,
                control.unsqueeze(1),
                intersect.unsqueeze(1),
            ),
            1,
        )
        feat = self.meta(torch.cat((feat, meta), 1))

        res = feat
        for i in range(len(self.fuse["ctr"])):
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre"):
                    k1 = int(key[3:])
                    # temp = index_add_naive(
                    #     temp,
                    #     self.fuse[key][i](feat[pre[k1][1]]),
                    #     pre[k1][0]
                    # )
                    temp.index_add_(
                        0,
                        pre[k1][0],
                        self.fuse[key][i](feat[pre[k1][1]]),
                    )
                if key.startswith("suc"):
                    k1 = int(key[3:])
                    # temp = index_add_naive(
                    #     temp,
                    #     self.fuse[key][i](feat[suc[k1][1]]),
                    #     suc[k1][0]
                    # )
                    temp.index_add_(
                        0,
                        suc[k1][0],
                        self.fuse[key][i](feat[suc[k1][1]]),
                    )

            # temp = index_add_naive(
            #     temp,
            #     self.fuse["left"][i](feat[left[1]]),
            #     left[0]
            # )
            temp.index_add_(
                0,
                left[0],
                self.fuse["left"][i](feat[left[1]]),
            )
            # temp = index_add_naive(
            #     temp,
            #     self.fuse["right"][i](feat[right[1]]),
            #     right[0]
            # )
            temp.index_add_(
                0,
                right[0],
                self.fuse["right"][i](feat[right[1]]),
            )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat, ctrs
