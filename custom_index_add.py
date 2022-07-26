import torch


def index_add_naive(dst, src, idx):
    def reduce_op(x_in):
        return torch.sum(x_in, dim=0).unsqueeze(0)

    index_range = torch.max(idx) + 1
    bucket = [[torch.zeros(1, dst.shape[1], device=dst.device)] for _ in range(index_range)]

    for i in range(len(src)):
        bucket[idx[i]].append(src[i: i + 1])

    for i in range(len(bucket)):
        bucket[i] = reduce_op(torch.cat(bucket[i], dim=0))

    bucket = torch.cat(bucket, dim=0)

    dst[:bucket.shape[0]] = dst[:bucket.shape[0]] + bucket

    return dst


updates = torch.rand(5, 2).cuda()
index = torch.tensor([0, 2, 3, 3, 3], dtype=torch.int64)
x = torch.zeros(10, 2).cuda()

print(index_add_naive(x, updates, index))
print(updates)
