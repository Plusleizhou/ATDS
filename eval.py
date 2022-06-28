from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from load_config import config
from model.Net import Net
from model.Net import Loss
from utils import PostProcess, gpu
from processed_data import ProcessedDataset, collate_fn
from utils import visualization, visualization_for_all_agents
import os
from train import load_prev_weights
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ATDS-Net Evaluating')
    parser.add_argument('--save_path', default='log-0', type=str, help='checkpoint path')
    parser.add_argument("--val_path", default="./data/features/sampled_val/")
    parser.add_argument("--use_goal", action="store_true", default=False, help="whether to use goal")
    parser.add_argument('--num_preds', default=30, type=int, help="the number of prediction frames")
    parser.add_argument('--devices', default='0', type=str, help='gpu devices for training')
    parser.add_argument('--viz', action="store_true", default=False, help='whether to visualize the prediction')
    args = parser.parse_args()

    # update config
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    model_name = args.save_path
    config["save_dir"] = os.path.join("results", model_name, "weights/")
    config["result"] = os.path.join("results", model_name, "result.txt")
    config["images"] = os.path.join("results", model_name, "images/")
    config["competition_files"] = os.path.join("results", model_name, "competition/")
    config["processed_val"] = args.val_path
    return args


def main():
    args = parse_args()
    dataset = ProcessedDataset(config["processed_val"], mode="val")
    val_dataloader = DataLoader(dataset,
                                batch_size=config['val_batch_size'],
                                num_workers=config['val_workers'],
                                shuffle=False,
                                collate_fn=collate_fn)
    net = Net(config).cuda()
    loss_net = Loss(config).cuda()
    net.eval()

    if len(os.listdir(config['save_dir'])) > 0:
        files = os.listdir(config['save_dir'])
        if 'net.pth' in files:
            files.remove('net.pth')
        files.sort(key=lambda x: float(x[:-4]))
        path = config['save_dir'] + files[-1]
        load_prev_weights(net, path)
        # net.load_state_dict(torch.load(path, map_location="cuda:0"))
        print('load weights from %s' % path)
    else:
        print('from beginning!')
        # exit()

    metrics = dict()
    loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Val", leave=True)
    loop.set_postfix_str(f'total_loss="???", minFDE="???"')
    for i, batch in loop:
        with torch.no_grad():
            out = net(batch)

            # use key point
            if args.use_goal:
                for k in range(len(out['reg'])):
                    out['reg'][k][:, :, -1] = out['key_points'][k][:, :, -1]

            loss_out = loss_net(out, batch)
            post_out = PostProcess(out, batch)  # 只记录了agent的pred，gt_pred, 和has_pred
            post_out.append(metrics, loss_out)  # 加入了全部agent的loss_out
            # metrics记录了所有轨迹，实时计算平均误差
            if (i + 1) % 50 == 0:
                val_out = post_out.display(metrics, args.num_preds)  # agent的指标
                if args.viz:
                    visualization_for_all_agents(out, batch, i, True, False)
                    visualization(out, batch, i, True, False)
                loop.set_postfix_str(f'total_loss={val_out["loss"]:.3f}, minFDE={val_out["fde"]:.3f}')
    val_out = post_out.display(metrics, args.num_preds)
    for k, v in val_out.items():
        print("{}: {:.4f}".format(k, v))
    loop.set_postfix_str(f'total_loss={val_out["loss"]:.3f}, minFDE={val_out["fde"]:.3f}')


if __name__ == '__main__':
    main()
