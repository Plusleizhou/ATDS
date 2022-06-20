import argparse
import errno
import math
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from preprocess import PreProcess

# add lib path
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

        self.columns = ["TIMESTAMP", "FRAMES", "TRACK_ID", "OBJECT_TYPE", "X", "Y", "V", "YAW", "EGO_RELATION"]

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

            if df_lane.shape[0] > 0:
                df = pd.concat([df_ob, df_lane], axis=0, copy=False, sort=False)
            else:
                df = df_ob
            if df.shape[0] <= 0:
                return

            file_name = os.path.basename(bag_path).split(".")[0] + "_" + str(observed_frame) + "_" + str(self.ego_id)
            if self.args.debug and self.args.viz:
                self.plot_scenario(df, file_name)

            if not self.args.debug and self.args.plus:
                save_dir = os.path.join(os.getcwd(), "./data/processed/", file_name + "_plus" + ".pkl")
                if not os.path.exists(save_dir):
                    df.to_pickle(save_dir)
                continue

            # convert Plus data to Argo data
            data, headers = self.convertor.process(file_name, df)

            if not self.args.debug and self.args.argo:
                if os.path.abspath(self.args.save_dir):
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
            for topic_name, msg_raw, ts in bag.read_messages(topics=self.topics, raw=True):
                ts = ts.to_sec()
                msg = decoder.decode(topic_name, msg_raw[1])

                if topic_name == "/prediction/obstacles":
                    obs_num += 1
                    # obstacle step 1: obtain info of ego
                    if len(odom) > 0:
                        obs_list.append(
                            [ts, obs_num, self.ego_id, self.det_type["AV"], odom[0], odom[1], odom[2], odom[3],
                             self.ego_relation["EGO"]])

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
                                         ego_relation])

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
                                         ego_relation])

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
                                         ego_relation])

                    # obstacle step 2: obtain info of surrounding agents
                    pred_obs = msg.prediction_obstacle
                    if not pred_obs:
                        continue
                    for pred_ob in pred_obs:
                        # perception obstacle
                        obs = pred_ob.perception_obstacle
                        v = math.sqrt(obs.motion.vx * obs.motion.vx + obs.motion.vy * obs.motion.vy)
                        obs_list.append(
                            [ts, obs_num, obs.id, obs.type, obs.motion.x, obs.motion.y, v, obs.motion.yaw,
                             pred_ob.ego_relation])

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


def main():
    args = get_args()
    data_extractor = PlusPreproc(args)
    msg_decoder = topic_utils.MessageDecoder()
    for root, _, files in os.walk(args.bag_dir):
        files_loop = tqdm(enumerate(files), desc="Process", total=len(files), leave=True)
        for i, file in files_loop:
            f = open(args.record_file, "a+")
            if not file.endswith("db"):
                print("wrong file extension name")
                continue
            bag_path = os.path.join(root, file)
            data_extractor.extract_data_from_file(bag_path, msg_decoder)
            f.write(file + "\n")
            f.close()


if __name__ == "__main__":
    main()
