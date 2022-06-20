from model.Net import Net
from load_config import config
from processed_data import ProcessedDataset, collate_fn
from thop import profile, clever_format
from torch.utils.data import DataLoader


if __name__ == '__main__':
    net = Net(config).cuda()
    dataset = ProcessedDataset(config["processed_val"])
    val_dataloader = DataLoader(dataset,
                                batch_size=1,
                                num_workers=config['val_workers'],
                                shuffle=False,
                                collate_fn=collate_fn)
    input = dict()
    for batch in val_dataloader:
        for key in batch.keys():
            input[key] = batch[key]
        break
    flops, params = profile(net, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
