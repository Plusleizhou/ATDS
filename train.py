import numpy as np
from tqdm import tqdm
import visdom
import random
import torch
from torch.utils.data import DataLoader
from load_config import config
import torch.optim as optim
import time
from model.Net import Net
from model.Net import Loss
from utils import PostProcess, gpu
from processed_data import ProcessedDataset, collate_fn
from utils import visualization, create_dirs, save_log
import os
import warnings

warnings.filterwarnings("ignore")


def load_prev_weights(net, path):
    pre_state = torch.load(path, map_location="cuda:0")
    state_dict = net.state_dict()
    loaded_modules = []
    for k, v in pre_state.items():
        k = k.replace('module.', '', 1)
        module = k.split('.')[0]
        if module in config["ignored_modules"]:
            continue
        elif k in state_dict.keys() and state_dict[k].shape == v.shape:
            state_dict[k] = v
            loaded_modules.append(k)
    net.load_state_dict(state_dict)
    print(f'loaded parameters {len(loaded_modules)}/{len(state_dict)}')


def vis_visualization(vis, out, epoch):
    y = [v for k, v in out.items()]
    win_names = [k for k, v in out.items()]
    for i in range(len(y)):
        vis.line(Y=np.array([y[i]]), X=np.array([epoch]), win=win_names[i],
                 opts=dict(title=win_names[i]), update=None if epoch == 0 else 'append')


def worker_init_fn(pid):
    np_seed = int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


def val(data_loader, net, loss_net, vis=visualization, post_process=PostProcess, rank=0):
    net.eval()
    metrics = dict()
    loop = tqdm(enumerate(data_loader), total=len(data_loader), desc="Val", leave=False, disable=rank)
    for i, batch in loop:
        with torch.no_grad():
            out = net(batch)
            loss_out = loss_net(out, batch)
            post_out = post_process(out, batch)
            post_out.append(metrics, loss_out)
            if (i + 1) % 50 == 0 and rank == 0:
                vis(out, batch, i, True, False)
    val_out = post_out.display(metrics)
    net.train()
    return val_out


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # set seed
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # mkdir
    create_dirs()

    # visualization
    vis = visdom.Visdom(server='http://127.0.0.1', port=8097)
    assert vis.check_connection()

    dataset = ProcessedDataset(config["processed_train"], mode="train")
    train_dataloader = DataLoader(dataset,
                                  batch_size=config['train_batch_size'],
                                  num_workers=config['train_workers'],
                                  shuffle=True,
                                  pin_memory=True,
                                  collate_fn=collate_fn,
                                  worker_init_fn=worker_init_fn)

    dataset = ProcessedDataset(config["processed_val"], mode="val")
    val_dataloader = DataLoader(dataset,
                                batch_size=config['val_batch_size'],
                                num_workers=config['val_workers'],
                                shuffle=False,
                                pin_memory=True,
                                collate_fn=collate_fn)
    net = Net(config).cuda()
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
        print('load weights from %s, start epoch: %d' % (path, start_epoch))
        # 低版本torch也可以加载
        # torch.save(net.state_dict(), "20.pth.tar", _use_new_zipfile_serialization=False)
    else:
        start_epoch = 0
        print('from beginning!')

    epoch_loop = tqdm(range(start_epoch, config['epoch']), leave=True)
    for epoch in epoch_loop:
        net.train()
        num_batches = len(train_dataloader)
        epoch_per_batch = 1.0 / num_batches

        post_out = PostProcess()
        metrics = dict()
        iter_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Train", leave=False)
        iter_loop.set_postfix_str(f'lr={optimizer.param_groups[0]["lr"]:.5f}, total_loss="???", minFDE="???"')
        for i, batch in iter_loop:
            epoch += epoch_per_batch
            epoch_loop.set_description(f'Process [{epoch:.3f}/{config["epoch"]}]')

            out = net(batch)
            loss_out = loss_net(out, batch)
            post_out = PostProcess(out, batch)  # 只记录了agent的pred，gt_pred, 和has_pred
            post_out.append(metrics, loss_out)  # 加入了全部agent的loss_out

            optimizer.zero_grad()
            loss_out['loss'].backward()
            optimizer.step()
            scheduler.step(epoch)

            if (i + 1) % (num_batches // config['num_display']) == 0:
                train_out = post_out.display(metrics)
                vis_visualization(vis, train_out, epoch)
                iter_loop.set_postfix_str(
                    f'lr={optimizer.param_groups[0]["lr"]:.5f}, total_loss={train_out["loss"]:.3f}, '
                    f'minFDE={train_out["fde"]:.3f}')

        torch.save(net.state_dict(), config['save_dir'] + str(round(epoch)) + '.pth')

        train_out = post_out.display(metrics)
        iter_loop.set_postfix_str(f'lr={optimizer.param_groups[0]["lr"]:.5f}, total_loss={train_out["loss"]:.3f}, '
                                  f'minFDE={train_out["fde"]:.3f}')
        save_log(epoch, train_out)
        if round(epoch) % config['num_val'] == 0:
            val_out = val(val_dataloader, net, loss_net)
            save_log(epoch, val_out, True)


if __name__ == '__main__':
    main()
