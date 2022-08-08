import torch
from torch import nn


def index_add_naive(dst, src, idx):
    for i in range(src.shape[0]):
        dst[idx[i]] += src[i]
    return dst


class Conv1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, ng=1, act=True):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=(int(kernel_size) - 1) // 2, stride=stride,
                              bias=False)
        self.norm = nn.GroupNorm(ng, n_out)
        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Linear(nn.Module):
    def __init__(self, n_in, n_out, ng=1, act=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(n_in, n_out, bias=False)
        self.norm = nn.GroupNorm(ng, n_out)
        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, ng=1, act=True):
        super(Res1d, self).__init__()
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.GroupNorm(ng, n_out)
        self.bn2 = nn.GroupNorm(ng, n_out)

        if stride != 1 or n_out != n_in:
            self.downsample = nn.Sequential(
                nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(ng, n_out))
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class LinearRes(nn.Module):
    def __init__(self, n_in, n_out, ng=1):
        super(LinearRes, self).__init__()
        self.linear1 = nn.Linear(n_in, n_out, bias=False)
        self.linear2 = nn.Linear(n_out, n_out, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = nn.GroupNorm(ng, n_out)
        self.norm2 = nn.GroupNorm(ng, n_out)

        if n_in != n_out:
            self.transform = nn.Sequential(
                nn.Linear(n_in, n_out, bias=False),
                nn.GroupNorm(ng, n_out))
        else:
            self.transform = None

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.transform is not None:
            out += self.transform(x)
        else:
            out += x

        out = self.relu(out)
        return out
