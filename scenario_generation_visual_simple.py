#!/usr/bin/env python

# Simple command-line tool to echo ros messages. See --help for usage.

import argparse
import errno
import functools
import logging
import math
import os
import time
import re
import sys
import csv


from google.protobuf import text_format
from pluspy import bag_utils, log_utils, topic_utils
from utils.config import DATA_DIR

import fastbag
import rosbag
import rospy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import OrderedDict

topics = ['/perception/lane_path',
          '/perception/obstacles',
          '/prediction/obstacles',
          '/navsat/odom']

nums_history = 20

ego_id = 0
filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')

KeyType = {
    'FRONT_LEFT_TIRE' : 0,
    'FRONT_RIGHT_TIRE' :1,
    'REAR_LEFT_TIRE' : 2,
    'REAR_RIGHT_TIRE' : 3,
}

DET_TYPE = {'DONTCARE' : 0,
'UNKNOWN' : 1,
'CAR' : 11     ,
'PEDESTRIAN' : 12 ,
'BICYCLE' : 14 ,
'VAN' : 15 ,
'BUS' : 16 ,
'TRUCK': 17 ,
'TRAM' : 18 ,
'MOTO' : 19,
'BARRIER' : 20 ,
'CONE'    : 21 ,
'MOVABLE_SIGN' : 23 ,
'LICENSE_PLATE' : 26 ,
'SUV' : 27  ,
'LIGHTTRUCK' : 28 ,
'TRAILER' : 29,
'AGENT' : 2,
'AV' : 3,
'LEFT_BOUNDARY':4,
'CENTER_LANE':5,
'RIGHT_BOUNDARY':6}

EgoRelation = {
    'EGO' : 0,
    'LEFT' : 1,
    'RIGHT' : 2,
    'NONE' : 3,
    'NOT_SET' : 4
}

COLUMNS = ['TIMESTAMP', 'FRAMES', 'TRACK_ID', 'OBJECT_TYPE', 'X', 'Y', 'V', 'YAW', 'IMAGE_CX', 'IMAGE_CY', 
             'EGO_RELATION', 'S_DIST_TO_EGO', 'L_DIST_TO_EGO','LEFT_L_TO_EGO','RIGHT_L_TO_EGO',
             'MODEL_DETECTED_LEADING','OBSTRUCTED_START_IMU_X','OBSTRUCTED_LAT_OFFSET']

S_LOWER_BOUND = -20
S_UPPER_BOUND = 170

cut_in_path = os.path.join(DATA_DIR, 'cut-in')
cut_out_path = os.path.join(DATA_DIR, 'cut-out')
lf_ego_path = os.path.join(DATA_DIR, 'lane_follow_ego')
lf_nbr_path = os.path.join(DATA_DIR, 'lane_follow_neighbour')

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

def plot_scenario(scenario):
    ts = scenario['TIMESTAMP'].unique()
    df_lanes = scenario[scenario['OBJECT_TYPE'] == 'LANE']
    df_lane_now = df_lanes[df_lanes['TIMESTAMP'] == ts[20]]
    df_obs = scenario[scenario['OBJECT_TYPE'] != 'LANE']
    obs_group = df_obs.groupby('TRACK_ID')
    lanes_group = df_lane_now.groupby('TRACK_ID')
    for k, v in lanes_group.groups.items():
        lane_group = lanes_group.get_group(k)
        plt.plot(lane_group['X'], lane_group['Y'], 'k-')

    for obs_id, v in obs_group.groups.items():
        obs_info = obs_group.get_group(obs_id)
        if obs_id == ego_id:
            plt.plot(obs_info['X'], obs_info['Y'], 'b-', label='ego')
            plt.scatter(obs_info.iloc[0,3], obs_info.iloc[0,4], marker='*')
            plt.scatter(obs_info.iloc[-1,3], obs_info.iloc[-1,4], marker='o')
        elif DET_TYPE['CONE'] in obs_info['OBJECT_TYPE'].values or DET_TYPE['UNKNOWN'] in obs_info['OBJECT_TYPE'].values:
            plt.scatter(obs_info['X'], obs_info['Y'], marker='*')
        elif DET_TYPE['CAR'] in obs_info['OBJECT_TYPE'].values:
            plt.scatter(obs_info['X'], obs_info['Y'], marker='o')
        elif DET_TYPE['TRUCK'] in obs_info['OBJECT_TYPE'].values:
            plt.scatter(obs_info['X'], obs_info['Y'], marker='v')
        elif DET_TYPE['SUV'] in obs_info['OBJECT_TYPE'].values:
            plt.scatter(obs_info['X'], obs_info['Y'], marker='8')
        elif DET_TYPE['LIGHTTRUCK'] in obs_info['OBJECT_TYPE'].values:
            plt.scatter(obs_info['X'], obs_info['Y'], marker='p')
        else:
            plt.plot(obs_info['X'], obs_info['Y'], 'rh', label='obs')
    plt.show()
    # for obs_id, pred_obs_info in pred_obs.items():
    #     plt.plot(pred_obs_info[0], pred_obs_info[1], 'go', label='pred')

def mkdir(subpath):
    if not os.path.isdir(subpath):
        os.makedirs(subpath)

def mkdir_for_scenario():
    mkdir(cut_in_path)
    mkdir(cut_out_path)
    mkdir(lf_ego_path)
    mkdir(lf_nbr_path)

def store_file(row, pd_obs,pd_lanes):
    gt_frame = row.frame_id
    gt_obs_id = row.obs_id
    gt_intention = row.intention
    gt_ego_lane_status = row.ego_lane_status

    begin_time = time.time()

    begin = gt_frame - nums_history + 1
    df_ob = pd_obs[pd_obs['FRAMES'].isin(range(begin,gt_frame+1))]

    A = df_ob.columns.get_loc('TRACK_ID')
    B = df_ob.columns.get_loc('OBJECT_TYPE')
    df_new = df_ob.values
    df_new[df_new[:,A] == gt_obs_id,B] = DET_TYPE['AGENT']
    df_ob = pd.DataFrame(df_new, columns=df_ob.columns, index=df_ob.index)
    df_lane = pd.DataFrame([])

    if gt_frame in pd_lanes.groups.keys():
        df_lane = pd_lanes.get_group(gt_frame)

    if df_lane.shape[0] > 0:
        df = pd.concat([df_ob, df_lane], axis=0, copy=False, sort=False)
    else:
        df = df_ob
    if df.shape[0] <= 0:
        return
    df_intention = pd.DataFrame(data=df.shape[0] * [[gt_ego_lane_status,gt_intention]],
                                index=df.index, columns=['ego_lane_status', 'intention'])
    df = pd.concat([df, df_intention], axis=1, copy=False, sort=False)

    if gt_ego_lane_status == 'cut_in':
        sub_path = cut_in_path
    elif gt_ego_lane_status == 'cut_out':
        sub_path = cut_out_path
    elif gt_ego_lane_status == 'ego':
        sub_path = lf_ego_path
    else:
        sub_path = lf_nbr_path
    file_name = sub_path + '/' + row.bag_name + '_obstacle_' + str(gt_obs_id) + '_frames_' + str(
        gt_frame) + '_intent_' + str(gt_intention) + '.pkl'
    if not os.path.exists(file_name):
        df.to_pickle(file_name)
    last = time.time()
    # print('one frame = ', last - begin_time)

def do_echo_from_file(bag_path, msg_decoder, bag_df):
    logging.debug('Opening bag {}...'.format(bag_path))
    mkdir_for_scenario()
    begin_time = time.time()
    with open_bag(bag_path) as bag:
        logging.debug('Bag file opened. Reading...')

        obs_list = []
        lane_list = []
        lane_msg = None
        odom = ()

        lane_num = 0
        odom_num = 0
        obs_num = -1
        for topic_name, msg_raw, ts in bag.read_messages(topics=topics,raw=True):
            # print('topic_name =', topic_name)
            ts = ts.to_sec()
            msg = msg_decoder.decode(topic_name, msg_raw[1])
            # print('ts =', ts, 'topic_name', topic_name)
            if topic_name == '/prediction/obstacles':
                obs_num += 1
                if len(odom) > 0:
                    obs_list.append([ts, obs_num, ego_id, DET_TYPE['AV'], odom[0], odom[1], odom[2], odom[3],
                    0, 0, 0, 0, 0, 1.2, -1.2, False, 0, 0])

                ego_lane_points, left_lane_points, right_lane_points = [], [], []
                if lane_msg is not None and len(lane_msg.lane) > 0:
                    ego_lane_id = lane_msg.ego_lane_id
                    ego_left_lane_id = lane_msg.ego_left_lane_id
                    ego_right_lane_id = lane_msg.ego_right_lane_id
                    for lane in lane_msg.lane:
                        lane_id = lane.lane_id
                        if lane_id not in [ego_lane_id, ego_left_lane_id, ego_right_lane_id]:
                            continue

                        center_len,lfb_len,rtb_len = 500,500,500
                        if len(lane.center_curve.segment) > 0:
                            center_len = len(lane.center_curve.segment[0].line_segment.point)
                        if len(lane.left_boundary.curve.segment) > 0:
                            lfb_len = len(lane.left_boundary.curve.segment[0].line_segment.point)
                        if len(lane.right_boundary.curve.segment) > 0:
                            rtb_len = len(lane.right_boundary.curve.segment[0].line_segment.point)
                        min_point_len = min(center_len,lfb_len,rtb_len)
                        image_center_len = len(lane.image_center_curve)
                        image_lfb_len = len(lane.image_left_boundary)
                        image_rtb_len = len(lane.image_right_boundary)
                        min_image_point_len = min(image_center_len,image_lfb_len,image_rtb_len)

                        lane_points = []
                        lane_image_points = []
                        if len(lane.center_curve.segment) > 0:
                            lane_points = lane.center_curve.segment[0].line_segment.point
                            lane_image_points = lane.image_center_curve
                            if center_len >= 1.5 * min_point_len:
                                lane_points = lane.center_curve.segment[0].line_segment.point[::2]
                            if image_center_len >= 1.5 * min_image_point_len:
                                lane_image_points = lane.image_center_curve[::2]
                            # print('min_point_len = ',min_point_len)
                            # print('min_image_point_len = ',min_image_point_len)

                        if lane_id == ego_lane_id:
                            ego_lane_points = lane_points
                            ego_min_point_len = min_point_len
                            ego_min_image_point_len = min_image_point_len
                        elif lane_id == ego_left_lane_id:
                            left_lane_points = lane_points
                            left_min_point_len = min_point_len
                            left_min_image_point_len = min_image_point_len
                        else:
                            right_lane_points = lane_points
                            right_min_point_len = min_point_len
                            right_min_image_point_len = min_image_point_len

                left_obstructed_start_imu_x = None
                left_obstructed_lat_offset = None
                right_obstructed_start_imu_x = None
                right_obstructed_lat_offset = None
                if lane_msg is not None and len(lane_msg.lane) > 0:
                    lanes_p0 = []
                    for lane in lane_msg.lane:
                        lane_id = lane.lane_id
                        if lane_id == ego_lane_id:
                            ego_relation = EgoRelation['EGO']
                            min_point_len = ego_min_point_len
                            min_image_point_len = ego_min_image_point_len
                            if not np.isnan(lane.left_boundary.obstructed_start_imu_x):
                                left_obstructed_start_imu_x = lane.left_boundary.obstructed_start_imu_x
                                left_obstructed_lat_offset = lane.left_boundary.obstructed_lat_offset
                            if not np.isnan(lane.right_boundary.obstructed_start_imu_x):
                                right_obstructed_start_imu_x = lane.right_boundary.obstructed_start_imu_x
                                right_obstructed_lat_offset = lane.right_boundary.obstructed_lat_offset
                        elif lane_id == ego_left_lane_id:
                            ego_relation = EgoRelation['LEFT']
                            min_point_len = left_min_point_len
                            min_image_point_len = left_min_image_point_len
                        elif lane_id == ego_right_lane_id:
                            ego_relation = EgoRelation['RIGHT']
                            min_point_len = right_min_point_len
                            min_image_point_len = right_min_image_point_len
                        else:
                            continue

                        for segment in lane.center_curve.segment:
                            lane_points = segment.line_segment.point
                            lane_image_points = lane.image_center_curve
                            if center_len >= 1.5 * min_point_len:
                                lane_points = segment.line_segment.point[::2]
                            if image_center_len >= 1.5 * min_image_point_len:
                                lane_image_points = lane.image_center_curve[::2]
                            for i in range(len(lane_points)):
                                point = lane_points[i]
                                image_point_x, image_point_y = 0, 0
                                if i < len(lane_image_points):
                                    image_point_x = lane_image_points[i].x
                                    image_point_y = lane_image_points[i].y

                                if len(ego_lane_points) > i:
                                    ego_left_l = ego_lane_points[i].left_l
                                    ego_right_l = -ego_lane_points[i].right_l
                                else:
                                    ego_left_l = 1.925
                                    ego_right_l = -1.925

                                if len(right_lane_points) > i:
                                    right_left_l = right_lane_points[i].left_l
                                else:
                                    right_left_l = -1.925

                                if len(left_lane_points) > i:
                                    left_right_l = left_lane_points[i].right_l
                                else:
                                    left_right_l = 1.925

                                if lane_id == ego_lane_id:
                                    l_dist_to_ego = 0
                                elif lane_id == ego_left_lane_id:
                                    l_dist_to_ego = ego_left_l + left_right_l
                                else:
                                    l_dist_to_ego = ego_right_l - right_left_l

                                lane_list.append([ts, obs_num, lane_id, DET_TYPE['CENTER_LANE'], point.x, point.y, 0, point.yaw, image_point_x, image_point_y,
                                ego_relation, np.nan, l_dist_to_ego, l_dist_to_ego, l_dist_to_ego, False, np.nan, np.nan])

                        for segment in lane.left_boundary.curve.segment:
                            lane_points = segment.line_segment.point
                            # remove deduplicated lanes
                            if (len(lane_points) == 0) or (len(lane_points) > 0 and (lane_points[0].x,lane_points[0].y) in lanes_p0):
                                continue
                            lanes_p0.append((lane_points[0].x,lane_points[0].y))
                            lane_image_points = lane.image_left_boundary
                            if len(segment.line_segment.point) >= 1.5 * min_point_len:
                                lane_points = segment.line_segment.point[::2]
                            if image_lfb_len >= 1.5 * min_image_point_len:
                                lane_image_points = lane.image_left_boundary[::2]
                            # print('len(lane_points) = ',len(lane_points))
                            # print('len(ego_lane_points) = ',len(ego_lane_points))
                            for i in range(len(lane_points)):
                                point = lane_points[i]
                                image_point_x, image_point_y = 0, 0
                                if i < len(lane_image_points):
                                    image_point_x = lane_image_points[i].x
                                    image_point_y = lane_image_points[i].y
                                
                                if len(ego_lane_points) > i:
                                    ego_left_l = ego_lane_points[i].left_l
                                    ego_right_l = -ego_lane_points[i].right_l
                                    ego_yaw = ego_lane_points[i].yaw
                                else:
                                    ego_left_l = 1.925
                                    ego_right_l = -1.925
                                    ego_yaw = 0.0
                                
                                if len(left_lane_points) > i:
                                    left_lane_width = left_lane_points[i].left_l + left_lane_points[i].right_l
                                    left_yaw = left_lane_points[i].yaw
                                else:
                                    left_lane_width = 3.85
                                    left_yaw = 0.0

                                if len(right_lane_points) > i:
                                    right_yaw = right_lane_points[i].yaw
                                else:
                                    right_yaw = 0.0

                                if lane_id == ego_lane_id:
                                    yaw = ego_yaw
                                    l_dist_to_ego = ego_left_l
                                elif lane_id == ego_right_lane_id:
                                    l_dist_to_ego = ego_right_l
                                    yaw = right_yaw
                                else:
                                    l_dist_to_ego = ego_left_l + left_lane_width
                                    yaw = left_yaw
                                lane_list.append([ts, obs_num, lane_id,  DET_TYPE['LEFT_BOUNDARY'], point.x, point.y, 0, yaw, image_point_x, image_point_y,
                                ego_relation, np.nan, l_dist_to_ego, l_dist_to_ego, l_dist_to_ego, False,lane.left_boundary.obstructed_start_imu_x, lane.left_boundary.obstructed_lat_offset])
                        
                        for segment in lane.right_boundary.curve.segment:
                            lane_points = segment.line_segment.point
                            if (len(lane_points) == 0) or (len(lane_points) > 0 and (lane_points[0].x,lane_points[0].y) in lanes_p0):
                                continue
                            lanes_p0.append((lane_points[0].x, lane_points[0].y))

                            lane_image_points = lane.image_right_boundary
                            if len(segment.line_segment.point) >= 1.5 * min_point_len:
                                lane_points = segment.line_segment.point[::2]
                            if image_rtb_len >= 1.5 * min_image_point_len:
                                lane_image_points = lane.image_right_boundary[::2]
                            for i in range(len(lane_points)):
                                point = lane_points[i]
                                image_point_x, image_point_y = 0, 0
                                if i < len(lane_image_points):
                                    image_point_x = lane_image_points[i].x
                                    image_point_y = lane_image_points[i].y

                                if len(ego_lane_points) > i:
                                    ego_left_l = ego_lane_points[i].left_l
                                    ego_right_l = -ego_lane_points[i].right_l
                                    ego_yaw = ego_lane_points[i].yaw
                                else:
                                    ego_left_l = 1.925
                                    ego_right_l = -1.925
                                    ego_yaw = 0.0

                                if len(right_lane_points) > i:
                                    right_lane_width = right_lane_points[i].left_l + right_lane_points[i].right_l
                                    right_yaw = right_lane_points[i].yaw
                                else:
                                    right_lane_width = 3.85
                                    right_yaw = 0.0

                                if len(left_lane_points) > i:
                                    left_yaw = left_lane_points[i].yaw
                                else:
                                    left_yaw = 0.0

                                if lane_id == ego_lane_id:
                                    l_dist_to_ego = ego_right_l
                                    yaw = ego_yaw
                                elif lane_id == ego_left_lane_id:
                                    l_dist_to_ego = ego_left_l
                                    yaw = left_yaw
                                else:
                                    l_dist_to_ego = ego_right_l - right_lane_width
                                    yaw = right_yaw
                                lane_list.append([ts, obs_num, lane_id,  DET_TYPE['RIGHT_BOUNDARY'], point.x, point.y, 0, yaw, image_point_x, image_point_y,
                                ego_relation, np.nan, l_dist_to_ego, l_dist_to_ego, l_dist_to_ego, False, lane.right_boundary.obstructed_start_imu_x, lane.right_boundary.obstructed_lat_offset])
                        
                pred_obs = msg.prediction_obstacle
                has_cone = False
                if not pred_obs:
                    continue
                
                max_left_obstructed_offset = np.nan
                max_right_obstructed_offset = np.nan
                if left_obstructed_start_imu_x != None or right_obstructed_start_imu_x != None:
                    left_obstructed_id = np.nan
                    min_left_obstructed_x = np.inf
                    right_obstructed_id = np.nan
                    min_right_obstructed_x = np.inf
                    for pred_ob in pred_obs:
                        # perception obstacle
                        obs = pred_ob.perception_obstacle
                        if left_obstructed_start_imu_x != None and pred_ob.ego_relation == EgoRelation['LEFT']:
                            if not np.isnan(pred_ob.s_to_ego) and abs(pred_ob.s_to_ego - left_obstructed_start_imu_x) < min_left_obstructed_x:
                                min_left_obstructed_x = abs(pred_ob.s_to_ego - left_obstructed_start_imu_x)
                                left_obstructed_id = obs.id
                        if right_obstructed_start_imu_x != None and pred_ob.ego_relation == EgoRelation['RIGHT']:
                            if not np.isnan(pred_ob.s_to_ego) and abs(pred_ob.s_to_ego - right_obstructed_start_imu_x) < min_right_obstructed_x:
                                min_right_obstructed_x = abs(pred_ob.s_to_ego - right_obstructed_start_imu_x)
                                right_obstructed_id = obs.id
                    if left_obstructed_start_imu_x != None and min_left_obstructed_x < 10:
                        max_left_obstructed_offset = left_obstructed_lat_offset
                    if right_obstructed_start_imu_x != None and min_right_obstructed_x < 10:
                        max_right_obstructed_offset = right_obstructed_lat_offset
                    # print("min_left_obstructed_x = ", min_left_obstructed_x, "min_right_obstructed_x = ", min_right_obstructed_x, )
                    # print("left_obstructed_start_imu_x = ", left_obstructed_start_imu_x, "right_obstructed_start_imu_x = ", right_obstructed_start_imu_x, )
                    # print("left_obstructed_id = ", left_obstructed_id, "right_obstructed_id = ", right_obstructed_id, )
                    # print("left_obstructed_lat_offset = ", left_obstructed_lat_offset, "right_obstructed_lat_offset = ", right_obstructed_lat_offset, )
                
                for pred_ob in pred_obs:
                    # perception obstacle
                    obs = pred_ob.perception_obstacle
                    v = math.sqrt(obs.motion.vx * obs.motion.vx + obs.motion.vy * obs.motion.vy)
                    if not np.isnan(max_left_obstructed_offset) and obs.id == left_obstructed_id:
                        obs_list.append([ts, obs_num, obs.id, obs.type, obs.motion.x, obs.motion.y, v, obs.motion.yaw, obs.image_cx, obs.image_cy,
                        pred_ob.ego_relation, pred_ob.s_to_ego, pred_ob.l_to_ego, pred_ob.left_l_to_ego, pred_ob.right_l_to_ego,
                        obs.model_detected_leading, left_obstructed_start_imu_x, max_left_obstructed_offset])
                    elif not np.isnan(max_right_obstructed_offset) and obs.id == right_obstructed_id:
                        obs_list.append([ts, obs_num, obs.id, obs.type, obs.motion.x, obs.motion.y, v, obs.motion.yaw, obs.image_cx, obs.image_cy,
                        pred_ob.ego_relation, pred_ob.s_to_ego, pred_ob.l_to_ego, pred_ob.left_l_to_ego, pred_ob.right_l_to_ego,
                        obs.model_detected_leading, right_obstructed_start_imu_x, max_right_obstructed_offset])
                    else:
                        obs_list.append([ts, obs_num, obs.id, obs.type, obs.motion.x, obs.motion.y, v, obs.motion.yaw, obs.image_cx, obs.image_cy,
                        pred_ob.ego_relation, pred_ob.s_to_ego, pred_ob.l_to_ego, pred_ob.left_l_to_ego, pred_ob.right_l_to_ego,
                        obs.model_detected_leading, np.nan, np.nan])
            # if obs_num > 500:
            #     break

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
                odom = (x,y,v,yaw)
                odom_num += 1
        read_time = time.time()
        print(" read bag time : ", read_time - begin_time)
        pd_obs = pd.DataFrame(data=obs_list, columns=COLUMNS)
        pb_lanes = pd.DataFrame(data=lane_list, columns=COLUMNS)
        # pb_lanes['OBJECT_TYPE'] = pb_lanes['OBJECT_TYPE'].astype(str)
        pb_lanes = pb_lanes.groupby('FRAMES')
        # pd_obs['OBJECT_TYPE'] = pd_obs['OBJECT_TYPE'].astype(str)
        last = time.time()
        print('size = ',bag_df.shape[0])
        bag_df.apply(store_file,args = (pd_obs,pb_lanes,),axis=1)
        finish = time.time()
        print(" store all bag time : ", finish - last)

def get_args(_args=None):
    parser = argparse.ArgumentParser(description='Echo PlusAI custom ros messages')
    parser.add_argument('-b', '--bag',
                        dest='bag_path',
                        metavar="BAGFILE",
                        type=str,
                        default= '/mnt/prediction/auto-update-data/20220101-20220301/test/bags',
                        help="Path to the bag file to read")

    parser.add_argument('-gt-file',
                        type=str,
                        default= '/mnt/prediction/auto-update-data/20220101-20220301/test/test_data.txt',
                        help="Path to the ground truth file to read")

    parser.add_argument('-record-file',
                        type=str,
                        default= '/home/richard.lee/prediction_offline/vectornet/record_visual_test_bag.txt',
                        help="Path to the ground truth file to read")    

    args = parser.parse_args(args=_args)
    args.read_bag = args.bag_path is not None

    return args


def main():
    args = get_args()

    gt_df = pd.read_csv(args.gt_file,sep='\t')
    bags_df = gt_df.groupby('bag_name')
    bag_names = bags_df.groups.keys()

    msg_decoder = topic_utils.MessageDecoder()
    for root, dirs, files in os.walk(args.bag_path):
        for file in files:
            f = open(args.record_file,'a+')
            records = f.readlines()
            if not file.endswith('db'):
                print('file')
                continue
            bag_name, extend = os.path.splitext(file)
            if (bag_name not in bag_names):
                print('bag_name {} not in ground truth file.'.format(bag_name))
                continue
            if ((bag_name + '\n') in records):
                print('bag_name {} already be extracted.'.format(bag_name))
                continue            
            bag_df = bags_df.get_group(bag_name)
            if bag_df.shape[0] < nums_history:
                continue
            bag_path = os.path.join(root,file)
            print('bag_path = ',bag_path)
            do_echo_from_file(bag_path, msg_decoder, bag_df)
            f.write(bag_name + '\n')
            print('bag_name {} is writed into record file.'.format(bag_name))
            f.close()

if __name__ == '__main__':
    import sys
    sys.path.append('/opt/plusai/lib/python/')
    main()