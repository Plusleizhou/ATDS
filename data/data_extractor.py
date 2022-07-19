import argparse
import errno
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
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

        self.ego_id = 0

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

        self.columns = ["TIMESTAMP", "FRAMES", "TRACK_ID", "OBJECT_TYPE", "X", "Y", "V", "YAW", "EGO_RELATION",
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

    def store_file(self, pd_obs, pb_lanes, bag_path):
        sample_loop = tqdm(range(0, pd_obs["FRAMES"].unique().max() - self.args.pred_len - self.args.obs_len + 1),
                           desc="Processing",
                           total=pd_obs["FRAMES"].unique().max() - self.args.pred_len - self.args.obs_len + 1,
                           leave=False)
        for begin in sample_loop:
            observed_frame = begin + self.args.obs_len - 1
            end = begin + self.args.obs_len + self.args.pred_len
            df_ob = pd_obs[pd_obs["FRAMES"].isin(range(begin, end))]

            track_id_index = df_ob.columns.get_loc("TRACK_ID")
            object_type_index = df_ob.columns.get_loc("OBJECT_TYPE")
            df_new = df_ob.values
            df_new[df_new[:, track_id_index] == self.ego_id, object_type_index] = self.det_type["AGENT"]
            df_ob = pd.DataFrame(df_new, columns=df_ob.columns, index=df_ob.index)
            df_lane = pd.DataFrame([])

            if observed_frame in pb_lanes.groups.keys():
                df_lane = pb_lanes.get_group(observed_frame)
                lane_cl = df_lane[df_lane["OBJECT_TYPE"] == self.det_type["CENTER_LANE"]]
                if lane_cl.shape[0] <= 0:
                    continue

            if df_lane.shape[0] > 0:
                df = pd.concat([df_ob, df_lane], axis=0, copy=False, sort=False)
            else:
                continue

            agent = df[df["OBJECT_TYPE"] == self.det_type["AGENT"]]
            if agent.shape[0] != self.args.obs_len + self.args.pred_len:
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
                data, headers = self.convertor.process(file_name, df)
                if np.all(data[0][-1] != 2) and self.args.ego_lead:
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
            obs_list, lane_list, lane_msg, odom = [], [], None, ()
            obs_num, lane_num, odom_num = -1, 0, 0
            ts_list = []
            for topic_name, msg_raw, ts in bag.read_messages(topics=self.topics, raw=True):
                ts = ts.to_sec()

                # remove duplicated data with the same timestamp
                if len(ts_list) > 0 and min(abs(np.asarray(ts_list) - ts)) < 1e-4:
                    continue
                else:
                    ts_list.append(ts)

                msg = decoder.decode(topic_name, msg_raw[1])

                if topic_name == "/prediction/obstacles":
                    obs_num += 1
                    # obstacle step 1: obtain info of ego
                    if len(odom) > 0:
                        obs_list.append(
                            [ts, obs_num, self.ego_id, self.det_type["AV"], odom[0], odom[1], odom[2], odom[3],
                             self.ego_relation["EGO"], False, None, self.ego_lane_presence["IN_EGOLANE"]])

                    # lane
                    if lane_msg is not None and len(lane_msg.lane) > 0:
                        ego_lane_id = lane_msg.ego_lane_id
                        ego_left_lane_id = lane_msg.ego_left_lane_id
                        ego_right_lane_id = lane_msg.ego_right_lane_id
                        for lane in lane_msg.lane:
                            lane_id = lane.lane_id
                            if lane_id not in [ego_lane_id, ego_left_lane_id, ego_right_lane_id]:
                                continue

                            center_len, lfb_len, rtb_len = 500, 500, 500
                            if len(lane.center_curve.segment) > 0:
                                center_len = len(lane.center_curve.segment[0].line_segment.point)
                            if len(lane.left_boundary.curve.segment) > 0:
                                lfb_len = len(lane.left_boundary.curve.segment[0].line_segment.point)
                            if len(lane.right_boundary.curve.segment) > 0:
                                rtb_len = len(lane.right_boundary.curve.segment[0].line_segment.point)
                            min_point_len = min(center_len, lfb_len, rtb_len)

                        lanes_p0 = []
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

                            for segment in lane.center_curve.segment:
                                lane_points = segment.line_segment.point
                                if center_len >= 1.5 * min_point_len:
                                    lane_points = segment.line_segment.point[::2]
                                for i in range(len(lane_points)):
                                    point = lane_points[i]
                                    lane_list.append(
                                        [ts, obs_num, lane_id, self.det_type["CENTER_LANE"], point.x, point.y, 0, 0,
                                         ego_relation, None, None, None])

                            for segment in lane.left_boundary.curve.segment:
                                lane_points = segment.line_segment.point
                                # remove deduplicated lanes
                                if (len(lane_points) == 0) or (
                                        len(lane_points) > 0 and (lane_points[0].x, lane_points[0].y) in lanes_p0):
                                    continue
                                lanes_p0.append((lane_points[0].x, lane_points[0].y))
                                if len(segment.line_segment.point) >= 1.5 * min_point_len:
                                    lane_points = segment.line_segment.point[::2]
                                for i in range(len(lane_points)):
                                    point = lane_points[i]
                                    lane_list.append(
                                        [ts, obs_num, lane_id, self.det_type["LEFT_BOUNDARY"], point.x, point.y, 0, 0,
                                         ego_relation, None, None, None])

                            for segment in lane.right_boundary.curve.segment:
                                lane_points = segment.line_segment.point
                                if (len(lane_points) == 0) or (
                                        len(lane_points) > 0 and (lane_points[0].x, lane_points[0].y) in lanes_p0):
                                    continue
                                lanes_p0.append((lane_points[0].x, lane_points[0].y))

                                if len(segment.line_segment.point) >= 1.5 * min_point_len:
                                    lane_points = segment.line_segment.point[::2]
                                for i in range(len(lane_points)):
                                    point = lane_points[i]
                                    lane_list.append(
                                        [ts, obs_num, lane_id, self.det_type["RIGHT_BOUNDARY"], point.x, point.y, 0, 0,
                                         ego_relation, None, None, None])

                    # obstacle step 2: obtain info of surrounding agents
                    pred_obs = msg.prediction_obstacle
                    if not pred_obs:
                        continue
                    for i, pred_ob in enumerate(pred_obs):
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
                        v = math.sqrt(obs.motion.vx * obs.motion.vx + obs.motion.vy * obs.motion.vy)
                        obs_list.append(
                            [ts, obs_num, obs.id, obs.type, obs.motion.x, obs.motion.y, v, obs.motion.yaw,
                             pred_ob.ego_relation, has_preds, pred_traj, obs.ego_lane_presence])

                if topic_name == "/perception/lane_path":
                    lane_msg = msg
                    lane_num += 1

                if topic_name == "/navsat/odom":
                    x = msg.pose.pose.position.x
                    y = msg.pose.pose.position.y
                    vx = msg.twist.twist.linear.x
                    vy = msg.twist.twist.linear.y
                    v = math.sqrt(vx * vx + vy * vy)
                    yaw = msg.pose.pose.orientation.z
                    odom = (x, y, v, yaw)
                    odom_num += 1
            pd_obs = pd.DataFrame(data=obs_list, columns=self.columns)
            pb_lanes = pd.DataFrame(data=lane_list, columns=self.columns)
            pb_lanes = pb_lanes.groupby("FRAMES")
            if pd_obs.values.shape[0] > 0:
                self.store_file(pd_obs, pb_lanes, bag_path)
            # exit(0)

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
                ax.scatter(obs_info.iloc[0, 4], obs_info.iloc[0, 5], marker="*")
                ax.scatter(obs_info.iloc[-1, 4], obs_info.iloc[-1, 5], marker="o")
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
        # plt.show()


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
