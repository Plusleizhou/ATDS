import argparse
import torch.distributed as dist
import numpy as np
import random
from tqdm import tqdm
import visdom
import torch
from torch.utils.data import DataLoader
from load_config import config
import torch.optim as optim
import time
from model.Net import Net
from model.Net import Loss
from utils import PostProcess, gpu
from train import val, vis_visualization
from processed_data import ProcessedDataset, collate_fn
from utils import create_dirs, save_log
import os
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ATDS-Net Training')
    parser.add_argument('--save_path', default='log-0', type=str, help='checkpoint path')
    parser.add_argument("--train_path", default="./data/features/train/")
    parser.add_argument("--val_path", default="./data/features/sampled_val/")
    parser.add_argument('--epoch', default=50, type=int, help='training epoch')
    parser.add_argument('--batch_size', default=32, type=int, help='the number of samples for one batch')
    parser.add_argument('--workers', default=0, type=int, help='the number of threads for loading data')
    parser.add_argument('--devices', default='0', type=str, help='gpu devices for training')
    parser.add_argument('--env', default='main', type=str, help='visdom env name')
    parser.add_argument('--record', action="store_true", default=False, help='whether to record the running log')
    args = parser.parse_args()

    # update config
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    config["epoch"] = args.epoch
    config["train_batch_size"] = args.batch_size
    config["train_workers"] = args.workers
    model_name = args.save_path
    config["save_dir"] = os.path.join("results", model_name, "weights/")
    config["result"] = os.path.join("results", model_name, "result.txt")
    config["images"] = os.path.join("results", model_name, "images/")
    config["competition_files"] = os.path.join("results", model_name, "competition/")
    config["processed_train"] = args.train_path
    config["processed_val"] = args.val_path
    return args


def load_prev_weights(net, path):
    pre_state = torch.load(path, map_location=torch.device('cpu'))
    state_dict = net.state_dict()
    loaded_modules = []
    for k, v in pre_state.items():
        module = k.split('.')[1]
        if module in config["ignored_modules"]:
            continue
        elif k in state_dict.keys() and state_dict[k].shape == v.shape:
            state_dict[k] = v
            loaded_modules.append(k)
    net.load_state_dict(state_dict)
    if dist.get_rank() == 0:
        print(f'loaded parameters {len(loaded_modules)}/{len(state_dict)}')


def worker_init_fn(pid):
    np_seed = dist.get_rank() % 2 * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    # set seed
    init_seeds(dist.get_rank() + 1)

    # mkdir
    if dist.get_rank() == 0:
        create_dirs()

    # visualization
    vis = visdom.Visdom(server='http://127.0.0.1', port=8097, env=args.env)
    assert vis.check_connection()

    dataset = ProcessedDataset(config["processed_train"], mode="train")
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset,
                                  batch_size=config['train_batch_size'],
                                  num_workers=config['train_workers'],
                                  collate_fn=collate_fn,
                                  pin_memory=True,
                                  sampler=train_sampler,
                                  worker_init_fn=worker_init_fn)
    dataset = ProcessedDataset(config["processed_val"], mode="val")
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    val_dataloader = DataLoader(dataset,
                                batch_size=config['val_batch_size'],
                                num_workers=config['val_workers'],
                                collate_fn=collate_fn,
                                pin_memory=True,
                                sampler=val_sampler)
    net = Net(config).cuda(local_rank)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], find_unused_parameters=True)

    loss_net = Loss(config).cuda()
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=config["lr"])
    else:
        optimizer = optim.AdamW(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    if config["scheduler"] == "CosineAnnWarm":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config["T_0"],
                                                                         T_mult=config["T_mult"],
                                                                         eta_min=max(1e-2 * config["lr"], 1e-6),
                                                                         last_epoch=-1)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["milestones"],
                                                         gamma=config["gamma"], last_epoch=-1)

    if len(os.listdir(config['save_dir'])) > 0:
        files = os.listdir(config['save_dir'])
        if 'net.pth' in files:
            files.remove('net.pth')
        files.sort(key=lambda x: float(x[:-4]))
        path = config['save_dir'] + files[-1]
        # net.load_state_dict(torch.load(path))
        load_prev_weights(net, path)
        start_epoch = int(float(files[-1][:-4]))
        if dist.get_rank() == 0:
            print('load weights from %s, start epoch: %d' % (path, start_epoch))
    else:
        start_epoch = 0
        if dist.get_rank() == 0:
            print('from beginning!')

    epoch_loop = tqdm(range(start_epoch, config['epoch']), leave=True, disable=dist.get_rank())
    for epoch in epoch_loop:
        net.train()
        num_batches = len(train_dataloader)
        epoch_per_batch = 1.0 / num_batches

        post_out = PostProcess()
        metrics = dict()
        iter_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                         desc="Train", leave=False, disable=dist.get_rank())
        iter_loop.set_postfix_str(f'lr={optimizer.param_groups[0]["lr"]:.5f}, total_loss="???", minFDE="???"')
        for i, batch in iter_loop:
            epoch += epoch_per_batch
            epoch_loop.set_description(f'Process [{epoch:.3f}/{config["epoch"]}]')

            out = net(batch)
            loss_out = loss_net(out, batch)

            if args.record:
                post_out = PostProcess(out, batch)  # 只记录了agent的pred，gt_pred, 和has_pred
                post_out.append(metrics, loss_out)  # 加入了全部agent的loss_out

            optimizer.zero_grad()
            loss_out['loss'].backward()
            optimizer.step()
            scheduler.step(epoch)

            if args.record and (i + 1) % (num_batches // config['num_display']) == 0 and dist.get_rank() == 0:
                train_out = post_out.display(metrics)
                vis_visualization(vis, train_out, epoch)
                iter_loop.set_postfix_str(
                    f'lr={optimizer.param_groups[0]["lr"]:.5f}, total_loss={train_out["loss"]:.3f}, '
                    f'minFDE={train_out["fde"]:.3f}')

        if dist.get_rank() == 0:
            torch.save(net.state_dict(), config['save_dir'] + str(round(epoch)) + '.pth')

        if args.record:
            train_out = post_out.display(metrics)
            iter_loop.set_postfix_str(f'lr={optimizer.param_groups[0]["lr"]:.5f}, total_loss={train_out["loss"]:.3f}, '
                                      f'minFDE={train_out["fde"]:.3f}')
            save_log(epoch, train_out)
        if round(epoch) % config['num_val'] == 0:
            val_out = val(val_dataloader, net, loss_net, dist.get_rank())
            save_log(epoch, val_out, True)


if __name__ == '__main__':
    # torchrun --nproc_per_node=1 multiple_train.py
    main()
