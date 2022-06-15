import argparse
import errno
import math
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import animation

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

TOPICS = ['/perception/lane_path',
          '/perception/obstacles',
          '/prediction/obstacles',
          '/navsat/odom']

COLUMNS = ['TIMESTAMP', 'FRAMES', 'TRACK_ID', 'OBJECT_TYPE', 'X', 'Y', 'V', 'YAW', 'EGO_RELATION']

EGO_ID = 0

DET_TYPE = {'DONTCARE': 0,
            'UNKNOWN': 1,
            'CAR': 11,
            'PEDESTRIAN': 12,
            'BICYCLE': 14,
            'VAN': 15,
            'BUS': 16,
            'TRUCK': 17,
            'TRAM': 18,
            'MOTO': 19,
            'BARRIER': 20,
            'CONE': 21,
            'MOVABLE_SIGN': 23,
            'LICENSE_PLATE': 26,
            'SUV': 27,
            'LIGHTTRUCK': 28,
            'TRAILER': 29,
            'AGENT': 2,
            'AV': 3,
            'LEFT_BOUNDARY': 4,
            'CENTER_LANE': 5,
            'RIGHT_BOUNDARY': 6}

EGO_RELATION = {
    'EGO': 0,
    'LEFT': 1,
    'RIGHT': 2,
    'NONE': 3,
    'NOT_SET': 4
}


def get_args():
    parser = argparse.ArgumentParser(description="Echo PlusAI custom ros messages")
    parser.add_argument('-v', '--vis',
                        action='store_true',
                        default=False,
                        help='whether to visualize the data')
    parser.add_argument('-d', '--vid',
                        action='store_true',
                        default=False,
                        help='whether to generate the video')
    parser.add_argument('-b', '--bag',
                        dest='bag_path',
                        type=str,
                        default="./data/snip_bag/",
                        help="Path to the bag file for extracting data")
    parser.add_argument('-record_file',
                        type=str,
                        default="./data/record_visual_test_bag.txt",
                        help="Path to the file for saved bag")
    args = parser.parse_args()
    return args


def open_bag(bag_path):
    if not os.path.isfile(bag_path):
        raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), bag_path)
    if bag_utils.is_ros_bag(bag_path):
        src_bag = rosbag.Bag(bag_path, 'r')
    elif fastbag.Reader.is_readable(bag_path):
        src_bag = fastbag.Reader(bag_path)
    else:
        raise IOError("Input bag {} cannot be parsed using either RosBag or FastBag!"
                      .format(bag_path))
    return src_bag


def generate_video(scenario, file, fig):
    fig.clf()
    fig.suptitle(file)
    ax = fig.add_subplot(111)

    ts = scenario["TIMESTAMP"].unique()
    lane_type = [DET_TYPE["LEFT_BOUNDARY"], DET_TYPE["CENTER_LANE"], DET_TYPE["RIGHT_BOUNDARY"]]
    df_lanes = scenario.query("OBJECT_TYPE in @lane_type")
    df_obs = scenario.query("OBJECT_TYPE not in @lane_type")
    obs_group = df_obs.groupby("TRACK_ID")
    lanes_group = df_lanes.groupby(["TRACK_ID", "OBJECT_TYPE"])
    for k in lanes_group.groups.keys():
        if k[1] == DET_TYPE["CENTER_LANE"]:
            lane_group = lanes_group.get_group(k)
            ax.plot(lane_group['X'].to_numpy(), lane_group['Y'].to_numpy())

    for obs_id, v in obs_group.groups.items():
        obs_info = obs_group.get_group(obs_id)
        if obs_id == EGO_ID:
            ax.plot(obs_info['X'].to_numpy(), obs_info['Y'].to_numpy(), 'b-', label='ego')
            ax.scatter(obs_info.iloc[0, 4], obs_info.iloc[0, 5], marker='*')
            ax.scatter(obs_info.iloc[-1, 4], obs_info.iloc[-1, 5], marker='o')
    plt.pause(0.01)
    return ax


def plot_scenario(scenario, file, fig=None):
    if fig is None:
        fig = plt.figure()
    fig.suptitle(file)
    ax = fig.add_subplot(111)

    ts = scenario["TIMESTAMP"].unique()
    lane_type = [DET_TYPE["LEFT_BOUNDARY"], DET_TYPE["CENTER_LANE"], DET_TYPE["RIGHT_BOUNDARY"]]
    df_lanes = scenario.query("OBJECT_TYPE in @lane_type")
    df_obs = scenario.query("OBJECT_TYPE not in @lane_type")
    obs_group = df_obs.groupby("TRACK_ID")
    lanes_group = df_lanes.groupby(["TRACK_ID", "OBJECT_TYPE"])
    for k in lanes_group.groups.keys():
        if k[1] == DET_TYPE["CENTER_LANE"]:
            lane_group = lanes_group.get_group(k)
            ax.plot(lane_group['X'].to_numpy(), lane_group['Y'].to_numpy())

    for obs_id, v in obs_group.groups.items():
        obs_info = obs_group.get_group(obs_id)
        if obs_id == EGO_ID:
            ax.plot(obs_info['X'].to_numpy(), obs_info['Y'].to_numpy(), 'b-', label='ego')
            ax.scatter(obs_info.iloc[0, 4], obs_info.iloc[0, 5], marker='*')
            ax.scatter(obs_info.iloc[-1, 4], obs_info.iloc[-1, 5], marker='o')
        elif DET_TYPE['CONE'] in obs_info['OBJECT_TYPE'].values or \
                DET_TYPE['UNKNOWN'] in obs_info['OBJECT_TYPE'].values:
            # plt.scatter(obs_info['X'].to_numpy(), obs_info['Y'].to_numpy(), marker='*')
            pass
        elif DET_TYPE['CAR'] in obs_info['OBJECT_TYPE'].values:
            ax.scatter(obs_info['X'].to_numpy(), obs_info['Y'].to_numpy(), marker='o')
        elif DET_TYPE['TRUCK'] in obs_info['OBJECT_TYPE'].values:
            ax.scatter(obs_info['X'].to_numpy(), obs_info['Y'].to_numpy(), marker='v')
        elif DET_TYPE['SUV'] in obs_info['OBJECT_TYPE'].values:
            ax.scatter(obs_info['X'].to_numpy(), obs_info['Y'].to_numpy(), marker='8')
        elif DET_TYPE['LIGHTTRUCK'] in obs_info['OBJECT_TYPE'].values:
            ax.scatter(obs_info['X'].to_numpy(), obs_info['Y'].to_numpy(), marker='p')
        else:
            # plt.plot(obs_info['X'].to_numpy(), obs_info['Y'].to_numpy(), 'rh', label='obs')
            pass
    plt.show()


def store_file(pd_obs, pb_lanes, bag_path):
    agent_id = 0
    observed_frame = 19
    num_history = 20
    num_future = 30

    sample_loop = tqdm(range(19, pd_obs["FRAMES"].unique().max()), desc="Processing",
                       total=pd_obs["FRAMES"].unique().max() - num_future - 19, leave=False)
    for observed_frame in sample_loop:
        begin = observed_frame - num_history + 1
        end = observed_frame + num_future + 1
        df_ob = pd_obs[pd_obs["FRAMES"].isin(range(begin, end))]

        track_id_index = df_ob.columns.get_loc("TRACK_ID")
        object_type_index = df_ob.columns.get_loc("OBJECT_TYPE")
        df_new = df_ob.values
        df_new[df_new[:, track_id_index] == agent_id, object_type_index] = DET_TYPE["AGENT"]
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

        file_name = os.path.join(os.getcwd(), "data/processed",
                                 os.path.basename(bag_path) + "_" + str(observed_frame) + "_" + str(agent_id) + ".pkl")
        if not os.path.exists(file_name):
            df.to_pickle(file_name)


def extract_data_from_file(bag_path, decoder):
    with open_bag(bag_path) as bag:
        obs_list, lane_list, lane_msg, odom = [], [], None, ()
        obs_num, lane_num, odom_num = -1, 0, 0
        for topic_name, msg_raw, ts in bag.read_messages(topics=TOPICS, raw=True):
            ts = ts.to_sec()
            msg = decoder.decode(topic_name, msg_raw[1])

            if topic_name == '/prediction/obstacles':
                obs_num += 1
                # obstacle step 1: obtain info of ego
                if len(odom) > 0:
                    obs_list.append(
                        [ts, obs_num, EGO_ID, DET_TYPE['AV'], odom[0], odom[1], odom[2], odom[3], EGO_RELATION["EGO"]])

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
                            ego_relation = EGO_RELATION['EGO']
                        elif lane_id == ego_left_lane_id:
                            ego_relation = EGO_RELATION['LEFT']
                        elif lane_id == ego_right_lane_id:
                            ego_relation = EGO_RELATION['RIGHT']
                        else:
                            continue

                        for segment in lane.center_curve.segment:
                            lane_points = segment.line_segment.point
                            if center_len >= 1.5 * min_point_len:
                                lane_points = segment.line_segment.point[::2]
                            for i in range(len(lane_points)):
                                point = lane_points[i]
                                lane_list.append(
                                    [ts, obs_num, lane_id, DET_TYPE['CENTER_LANE'], point.x, point.y, 0, 0,
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
                                    [ts, obs_num, lane_id, DET_TYPE['LEFT_BOUNDARY'], point.x, point.y, 0, 0,
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
                                    [ts, obs_num, lane_id, DET_TYPE['RIGHT_BOUNDARY'], point.x, point.y, 0, 0,
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

            if topic_name == '/perception/lane_path':
                lane_msg = msg
                lane_num += 1

            if topic_name == '/navsat/odom':
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                vx = msg.twist.twist.linear.x
                vy = msg.twist.twist.linear.y
                v = math.sqrt(vx * vx + vy * vy)
                yaw = msg.pose.pose.orientation.z
                odom = (x, y, v, yaw)
                odom_num += 1
        pd_obs = pd.DataFrame(data=obs_list, columns=COLUMNS)
        pb_lanes = pd.DataFrame(data=lane_list, columns=COLUMNS)
        pb_lanes = pb_lanes.groupby("FRAMES")
        store_file(pd_obs, pb_lanes, bag_path)
        # exit(0)


def main():
    args = get_args()
    if not args.vis and not args.vid:
        msg_decoder = topic_utils.MessageDecoder()
        for root, _, files in os.walk(args.bag_path):
            files_loop = tqdm(enumerate(files), desc="Process", total=len(files), leave=True)
            for i, file in files_loop:
                f = open(args.record_file, 'a+')
                if not file.endswith("db"):
                    print("wrong file extension name")
                    continue
                bag_path = os.path.join(root, file)
                extract_data_from_file(bag_path, msg_decoder)
                f.write(file + "\n")
                f.close()
    else:
        dir_path = os.path.join(os.getcwd(), 'data/processed')
        files = sorted(os.listdir(dir_path), key=lambda x: int(x.split('.')[1].split('_')[1]) +
                                                      len(os.listdir(dir_path)) * int(x.split('.')[1].split('_')[2]))
        if args.vis:
            for file in files:
                scenario = pd.read_pickle(os.path.join(dir_path, file))
                plot_scenario(scenario, file)
        elif args.vid:
            figure = plt.figure()
            plt.ion()
            for file in files:
                scenario = pd.read_pickle(os.path.join(dir_path, file))
                generate_video(scenario, file, figure)
            plt.ioff()
            plt.show()


if __name__ == "__main__":
    main()
