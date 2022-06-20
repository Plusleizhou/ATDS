import os


config = dict()
model_name = "log-0"

# net
config["n_agent"] = 128
config["n_map"] = 128

config["num_mods"] = 6
config["num_preds"] = 30
config["num_obs"] = 20
config["agent2map_dist"] = 7.0
config["map2agent_dist"] = 15.0
config["agent2agent_dist"] = 100.0
config["key_points"] = [9, 19]

config["cls_th"] = 10.0
config["cls_ignore"] = 0.2
config["mgn"] = 0.2

config["cls_coef"] = 1.0
config["reg_coef"] = 1.0
config["key_points_coef"] = 1.0

config["agent_extractor"] = 2  # 1: conv1d at last step, 2: self-attention
config["num_scales"] = 6

# weights
config["ignored_modules"] = []

# Processed Dataset
config["processed_train"] = "./data/features/train/"
config["processed_val"] = "./data/features/val/"
config["processed_test"] = "./data/features/test/"

# train
config["scaling_ratio"] = [0.8, 1.25]  # set [1, 1] to stop
config["past_motion_dropout"] = 11  # set 1 to stop
config["flip"] = 0.2  # set 0 to stop

config["epoch"] = 50
config["lr"] = 0.001
config["optimizer"] = "Adam"  # "AdamW"
config["weight_decay"] = 0.005
config["scheduler"] = "MultiStepLR"  # "CosineAnnWarm"
config["T_0"] = 2
config["T_mult"] = 1
config["milestones"] = [30, 48]
config["gamma"] = 0.1
config["train_batch_size"] = 4
config["train_workers"] = 0

config["num_display"] = 10
config["num_val"] = 2
config["num_save"] = 2
config["save_dir"] = os.path.join("results", model_name, "weights/")
config["result"] = os.path.join("results", model_name, "result.txt")
config["images"] = os.path.join("results", model_name, "images/")

# val
config["val_batch_size"] = 32
config["val_workers"] = config["train_workers"]

# test
config["test_batch_size"] = 32
config["test_workers"] = config["train_workers"]
config["competition_files"] = os.path.join("results", model_name, "competition/")
