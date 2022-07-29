import argparse
import errno
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
import multiprocessing
from matplotlib import pyplot as plt
from preprocess import PreProcess

# add lib path
if os.path.exists(os.path.join(os.getcwd(), "lib/python")) \
        and os.path.exists("/opt/ros/noetic/lib/python3/dist-packages/"):
    LIB_PATH = {
        "ros_path": "/opt/ros/noetic/lib/python3/dist-packages/",
        "plus_path": os.path.join(os.getcwd(), "lib/python")
    }
    for key in LIB_PATH.keys():
        sys.path.append(LIB_PATH[key])

import rosbag
import fastbag
from pluspy import bag_utils, topic_utils


def get_args():
    parser = argparse.ArgumentParser(description="Echo PlusAI custom ros messages")
    parser.add_argument("-b", "--bag_dir",
                        type=str,
                        default="./data/snip_bag/",
                        help="Path to the bag file for extracting data")
    parser.add_argument("-f", "--record_file",
                        type=str,
                        default="./data/record_visual_test_bag.txt",
                        help="Path to the file for saved bag")
    parser.add_argument("-s", "--save_dir",
                        default="./data/features/",
                        type=str,
                        help="Path where the computed features are saved")
    parser.add_argument("-m", "--mode",
                        required=True,
                        type=str,
                        help="train/val/test")
    parser.add_argument("--ego_lead",
                        default=False,
                        action="store_true",
                        help="Build dataset for ego lead")
    parser.add_argument("--plus",
                        default=False,
                        action="store_true",
                        help="Save data as plus format")
    parser.add_argument("--argo",
                        default=False,
                        action="store_true",
                        help="Save data as argo format")
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    parser.add_argument("--debug",
                        default=False,
                        action="store_true",
                        help="If true, debug mode.")
    parser.add_argument("--viz",
                        default=False,
                        action="store_true",
                        help="If true, viz.")

    args = parser.parse_args()
    return args


class PlusPreproc(object):
    def __init__(self, args):
        super(PlusPreproc, self).__init__()
        self.args = args
        self.convertor = PreProcess(args)

        self.ego_id = -1

        self.topics = [
            "/perception/lane_path",
            "/perception/obstacles",
            "/prediction/obstacles",
            "/navsat/odom"
        ]

        self.ego_relation = {
            "EGO": 0,
            "LEFT": 1,
            "RIGHT": 2,
            "NONE": 3,
            "NOT_SET": 4
        }

        self.ego_lane_presence = {
            "NOT_IN_EGOLANE": 0,
            "IN_EGOLANE": 1,
            "IN_EGOLANE_LEAD_VEHICLE": 2,
            "IN_RIGHT_SHOULDER": 3,
            "IN_RIGHT_SHOULDER_LEAD_VEHICLE": 4,
            "IN_LEFT_SHOULDER": 5,
            "IN_LEFT_SHOULDER_LEAD_VEHICLE": 6,
            "ON_ROAD": 7,
        }

        self.columns = ["TIMESTAMP", "TRACK_ID", "OBJECT_TYPE", "X", "Y", "VX", "VY", "YAW", "EGO_RELATION",
                        "HAS_PREDS", "PRED_TRAJ", "EGO_LANE_PRESENCE"]

        self.det_type = {
            "DONTCARE": 0,
            "UNKNOWN": 1,
            "CAR": 11,
            "PEDESTRIAN": 12,
            "BICYCLE": 14,
            "VAN": 15,
            "BUS": 16,
            "TRUCK": 17,
            "TRAM": 18,
            "MOTO": 19,
            "BARRIER": 20,
            "CONE": 21,
            "MOVABLE_SIGN": 23,
            "LICENSE_PLATE": 26,
            "SUV": 27,
            "LIGHTTRUCK": 28,
            "TRAILER": 29,
            "AGENT": 2,
            "AV": 3,
            "LEFT_BOUNDARY": 4,
            "CENTER_LANE": 5,
            "RIGHT_BOUNDARY": 6
        }

    def store_file(self, pd_ego, pd_obs, pd_lanes, bag_path):
        sample_loop = tqdm(range(pd_ego["TIMESTAMP"].unique().min(),
                                 pd_ego["TIMESTAMP"].unique().max() - self.args.pred_len - self.args.obs_len + 1),
                           desc="Processing",
                           total=pd_obs["TIMESTAMP"].unique().max() - pd_ego["TIMESTAMP"].unique().min()
                                 - self.args.pred_len - self.args.obs_len + 1,
                           leave=False)
        for begin in sample_loop:
            observed_frame = begin + self.args.obs_len - 1
            end = begin + self.args.obs_len + self.args.pred_len

            df_obs = pd_obs[pd_obs["TIMESTAMP"].isin(range(begin, end))]
            df_ego = pd_ego[pd_ego["TIMESTAMP"].isin(range(begin, end))]

            ts_ego = np.array(df_ego["TIMESTAMP"].values).astype(np.int64)
            if np.all(ts_ego > observed_frame) or observed_frame not in ts_ego:
                continue

            df_lane = pd.DataFrame([])
            if observed_frame in pd_lanes.groups.keys():
                df_lane = pd_lanes.get_group(observed_frame)
                lane_cl = df_lane[df_lane["OBJECT_TYPE"] == self.det_type["CENTER_LANE"]]
                if lane_cl.shape[0] <= 0:
                    continue
            if df_lane.shape[0] > 0:
                df = pd.concat([df_ego, df_obs, df_lane], axis=0, copy=False, sort=False)
            else:
                continue

            file_name = os.path.basename(bag_path).split(".")[0] + "_" + str(observed_frame) + "_" + str(self.ego_id)
            if self.args.debug and self.args.viz:
                self.plot_scenario(df, file_name)

            if not self.args.debug and self.args.plus:
                save_dir = os.path.join(os.getcwd(), "./data/processed/", file_name + "_plus" + ".pkl")
                if not os.path.exists(save_dir):
                    df.to_pickle(save_dir)
                continue

            if not self.args.debug and self.args.argo:
                # convert Plus data to Argo data
                data, headers = self.convertor.process(file_name, df, begin)
                if np.all(data[0][-2] != 2) and self.args.ego_lead:
                    continue
                if os.path.isabs(self.args.save_dir):
                    save_dir = os.path.join(self.args.save_dir, self.args.mode, file_name + "_argo" + ".pkl")
                else:
                    save_dir = os.path.join(os.getcwd(),
                                            self.args.save_dir, self.args.mode, file_name + "_argo" + ".pkl")
                if not os.path.exists(save_dir):
                    data_df = pd.DataFrame(data, columns=headers)
                    data_df.to_pickle(save_dir)

    def extract_data_from_file(self, bag_path, decoder):
        with self.open_bag(bag_path) as bag:
            obs_dict, lane_dict, ego_dict = defaultdict(), defaultdict(), defaultdict()
            for topic_name, msg_raw, ts in bag.read_messages(topics=self.topics, raw=True):
                ts = int(round(ts.to_sec() * 10))
                msg = decoder.decode(topic_name, msg_raw[1])

                if topic_name == "/prediction/obstacles":
                    pred_obs = msg.prediction_obstacle
                    if not pred_obs:
                        continue
                    obs_list = []
                    for pred_ob in pred_obs:
                        # trajectory
                        pred = pred_ob.trajectory
                        if len(pred) == 1:
                            has_preds = True
                            pred_traj = np.zeros((len(pred[0].trajectory_point[1:]), 2))
                            for k, loc in enumerate(pred[0].trajectory_point[1:]):
                                pred_traj[k, 0] = loc.x
                                pred_traj[k, 1] = loc.y
                        else:
                            has_preds = False
                            pred_traj = None
                        # perception obstacle
                        obs = pred_ob.perception_obstacle
                        if np.isnan(obs.motion.x) or np.isnan(obs.motion.y) \
                                or np.isnan(obs.motion.vx) or np.isnan(obs.motion.vy) or np.isnan(obs.motion.yaw):
                            continue
                        obs_list.append(
                            [ts, obs.id, obs.type, obs.motion.x, obs.motion.y, obs.motion.vx, obs.motion.vy,
                             obs.motion.yaw, pred_ob.ego_relation, has_preds, pred_traj, obs.ego_lane_presence])
                    obs_dict[ts] = obs_list

                if topic_name == "/perception/lane_path":
                    lane_msg = msg
                    if lane_msg is not None and len(lane_msg.lane) > 0:
                        ego_lane_id = lane_msg.ego_lane_id
                        ego_left_lane_id = lane_msg.ego_left_lane_id
                        ego_right_lane_id = lane_msg.ego_right_lane_id
                        lane_list = []
                        for lane in lane_msg.lane:
                            lane_id = lane.lane_id
                            if lane_id == ego_lane_id:
                                ego_relation = self.ego_relation["EGO"]
                            elif lane_id == ego_left_lane_id:
                                ego_relation = self.ego_relation["LEFT"]
                            elif lane_id == ego_right_lane_id:
                                ego_relation = self.ego_relation["RIGHT"]
                            else:
                                continue

                            lane_point_list = []
                            for segment in lane.center_curve.segment:
                                for point in segment.line_segment.point:
                                    if np.isnan(point.x) or np.isnan(point.y) or np.isnan(point.yaw):
                                        continue
                                    lane_point_list.append(
                                        [ts, lane_id, self.det_type["CENTER_LANE"], point.x, point.y, 0, 0, point.yaw,
                                         ego_relation, None, None, None])
                            if len(lane_point_list) > 0:
                                lane_list.append(lane_point_list)
                        if len(lane_list) != 0:
                            lane_dict[ts] = lane_list

                if topic_name == "/navsat/odom":
                    x = msg.pose.pose.position.x
                    y = msg.pose.pose.position.y
                    vx = msg.twist.twist.linear.x
                    vy = msg.twist.twist.linear.y
                    yaw = msg.pose.pose.orientation.z
                    odom = [ts, self.ego_id, self.det_type["AV"], x, y, vx, vy, yaw,
                            self.ego_relation["EGO"], False, None, self.ego_lane_presence["IN_EGOLANE"]]
                    ego_dict[ts] = odom
            ego_list, obs_list, map_list = self.gather(ego_dict, obs_dict, lane_dict)
            pd_ego = pd.DataFrame(data=ego_list, columns=self.columns)
            pd_obs = pd.DataFrame(data=obs_list, columns=self.columns)
            pd_lanes = pd.DataFrame(data=map_list, columns=self.columns)
            pd_lanes = pd_lanes.groupby("TIMESTAMP")
            if pd_ego.values.shape[0] > 0:
                self.store_file(pd_ego, pd_obs, pd_lanes, bag_path)
            # exit(0)

    @staticmethod
    def gather(ego_dict, obs_dict, lane_dict):
        ego_list, agent_list, map_list = [], [], []
        for ts in sorted(ego_dict.keys()):
            ego_list.append(ego_dict[ts])
        for ts in sorted(obs_dict.keys()):
            for obs in obs_dict[ts]:
                agent_list.append(obs)
        for ts in sorted(lane_dict.keys()):
            for lane in lane_dict[ts]:
                for point in lane:
                    map_list.append(point)
        return ego_list, agent_list, map_list

    @staticmethod
    def open_bag(bag_path):
        if not os.path.isfile(bag_path):
            raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), bag_path)
        if bag_utils.is_ros_bag(bag_path):
            src_bag = rosbag.Bag(bag_path, "r")
        elif fastbag.Reader.is_readable(bag_path):
            src_bag = fastbag.Reader(bag_path)
        else:
            raise IOError("Input bag {} cannot be parsed using either RosBag or FastBag!"
                          .format(bag_path))
        return src_bag

    def plot_scenario(self, scenario, file_name):
        _, ax = plt.subplots(figsize=(10, 10))
        ax.axis("equal")
        ax.set_title(file_name)

        ts = scenario["TIMESTAMP"].unique()
        lane_type = [self.det_type["LEFT_BOUNDARY"], self.det_type["CENTER_LANE"], self.det_type["RIGHT_BOUNDARY"]]
        df_lanes = scenario.query("OBJECT_TYPE in @lane_type")
        df_obs = scenario.query("OBJECT_TYPE not in @lane_type")
        obs_group = df_obs.groupby("TRACK_ID")
        lanes_group = df_lanes.groupby(["TRACK_ID", "OBJECT_TYPE"])
        for k in lanes_group.groups.keys():
            if k[1] == self.det_type["CENTER_LANE"]:
                lane_group = lanes_group.get_group(k)
                ax.plot(lane_group["X"].to_numpy(), lane_group["Y"].to_numpy())

        for obs_id, v in obs_group.groups.items():
            obs_info = obs_group.get_group(obs_id)
            if obs_id == self.ego_id:
                ax.plot(obs_info["X"].to_numpy(), obs_info["Y"].to_numpy(), "b-", label="ego")
                ax.scatter(obs_info["X"].to_numpy()[0], obs_info["Y"].to_numpy()[0], marker="*")
                ax.scatter(obs_info["X"].to_numpy()[-1], obs_info["Y"].to_numpy()[-1], marker="o")
            elif self.det_type["CONE"] in obs_info["OBJECT_TYPE"].values or \
                    self.det_type["UNKNOWN"] in obs_info["OBJECT_TYPE"].values:
                # plt.scatter(obs_info["X"].to_numpy(), obs_info["Y"].to_numpy(), marker="*")
                pass
            elif self.det_type["CAR"] in obs_info["OBJECT_TYPE"].values:
                ax.scatter(obs_info["X"].to_numpy(), obs_info["Y"].to_numpy(), marker="o")
            elif self.det_type["TRUCK"] in obs_info["OBJECT_TYPE"].values:
                ax.scatter(obs_info["X"].to_numpy(), obs_info["Y"].to_numpy(), marker="v")
            elif self.det_type["SUV"] in obs_info["OBJECT_TYPE"].values:
                ax.scatter(obs_info["X"].to_numpy(), obs_info["Y"].to_numpy(), marker="8")
            elif self.det_type["LIGHTTRUCK"] in obs_info["OBJECT_TYPE"].values:
                ax.scatter(obs_info["X"].to_numpy(), obs_info["Y"].to_numpy(), marker="p")
            else:
                # plt.plot(obs_info["X"].to_numpy(), obs_info["Y"].to_numpy(), "rh", label="obs")
                pass
        plt.show()


def load_bag_save_features(args, start_idx, batch_size, files):
    data_extractor = PlusPreproc(args)
    msg_decoder = topic_utils.MessageDecoder()

    for i, file in enumerate(files[start_idx: start_idx + batch_size]):
        if not file.endswith("db"):
            print("wrong file extension name")
            continue

        with open(args.record_file, "rt") as f_reader:
            records = [x.strip() for x in f_reader.readlines()]
        if args.mode + "_" + file in records:
            print("bag_name {} already be extracted".format(file))
            continue

        bag_path = os.path.join(args.bag_dir, file)
        data_extractor.extract_data_from_file(bag_path, msg_decoder)

        with open(args.record_file, "a") as f_writer:
            f_writer.write(args.mode + "_" + file + "\n")

    print("Finish computing {} - {}".format(start_idx, start_idx + batch_size))


def main():
    start = time.time()
    args = get_args()

    files = os.listdir(args.bag_dir)
    with open(args.record_file, "rt") as f_reader:
        saved_files = [x.strip().replace(args.mode + "_", "", 1) for x in f_reader.readlines()]

    # remove duplicated bags
    for f in saved_files:
        if f in files:
            print("bag_name {} already be extracted".format(f))
            files.remove(f)

    num_files = len(files)
    print("Num of files: ", num_files)

    n_proc = multiprocessing.cpu_count() - 2 if not args.debug else 1

    batch_size = np.max([int(np.ceil(num_files / float(n_proc))), 1])
    print('n_proc: {}, batch_size: {}'.format(n_proc, batch_size))

    Parallel(n_jobs=n_proc)(delayed(load_bag_save_features)(args, i, batch_size, files)
                            for i in range(0, num_files, batch_size))

    print("Preprocess for {} set completed in {} minutes".format(args.mode, (time.time() - start) / 60.0))


if __name__ == "__main__":
    main()
