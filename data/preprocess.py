import os
import copy
import argparse
import time
import multiprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from scipy import sparse, spatial
from tqdm import tqdm
from joblib import Parallel, delayed


class PreProcess(object):
    def __init__(self, args):
        super(PreProcess, self).__init__()
        self.args = args
        self.ls_cross_dist = 6.0
        self.ls_num_scales = 6
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
        self.ego_relation = {
            "EGO": 0,
            "LEFT": 1,
            "RIGHT": 2,
            "NONE": 3,
            "NOT_SET": 4
        }

    def process(self, seq_id, df, start_frame):
        # get trajectories
        ts, trajs, pad_flags, orig, rot, \
            pred_trajs, has_preds, ego_lane_presence, agent_ids = self.get_trajectories(df, start_frame)

        # build lane graph
        graph = self.get_lane_graph(df, orig, rot)

        # convert to np.float32
        orig = orig.astype(np.float32)
        rot = rot.astype(np.float32)

        # save data
        data = [[seq_id, orig, rot, ts, trajs, pad_flags, graph, has_preds, pred_trajs, ego_lane_presence, agent_ids]]
        headers = ["SEQ_ID", "ORIG", "ROT", "TIMESTAMP", "TRAJS", "PAD_FLAGS", "GRAPH", "HAS_PREDS", "PRED_TRAJS",
                   "EGO_LANE_PRESENCE", "AGENT_IDS"]

        # for debug
        if self.args.debug and self.args.viz:
            _, ax = plt.subplots(figsize=(10, 10))
            ax.axis("equal")
            vis_map = True  # whether to visualize map with trajectories
            self.plot_trajs(ax, trajs[:, :, :2], pad_flags, orig, rot, vis_map=vis_map)
            self.plot_lane_graph(ax, df, graph, orig, rot, vis_map=vis_map)
            ax.set_title(seq_id)
            plt.show()

        return data, headers

    def get_trajectory(self, df_agent, ts_agent, ts, t_obs, trajs, pad_flags, pred_trajs, has_preds, ego_lane_presence):
        df_agent_obs = df_agent[df_agent["TIMESTAMP"] == t_obs]
        pred_traj = df_agent_obs["PRED_TRAJ"].values[0]
        if np.all(df_agent_obs["HAS_PREDS"].values) and pred_traj.shape[0] >= 30:
            pred_trajs.append(pred_traj)
            has_preds.append(True)
        else:
            has_preds.append(False)
        ego_lane_presence.append(df_agent_obs["EGO_LANE_PRESENCE"].values[0])

        traj = np.stack((df_agent["X"].values, df_agent["Y"].values,
                         df_agent["VX"].values, df_agent["VY"].values, df_agent["YAW"].values), axis=1)
        _, ids, _ = np.intersect1d(ts, ts_agent, return_indices=True)
        padded = np.zeros_like(ts)
        padded[ids] = 1

        traj_pad = np.full((ts.shape[0], traj.shape[1]), None)
        traj_pad[ids] = traj
        traj_pad = self.padding_traj_nn(traj_pad)
        assert np.all(traj_pad[ids] == traj), "Padding error"

        traj = np.stack((traj_pad[:, 0], traj_pad[:, 1], traj_pad[:, 2], traj_pad[:, 3], traj_pad[:, 4]), axis=1)
        traj[:, :5] = traj[:, :5]
        trajs.append(traj)
        pad_flags.append(padded)

    def get_trajectories(self, df, start_frame):
        all_type = [self.det_type[k] for k in (set(self.det_type.keys()) - {"LEFT_BOUNDARY", "CENTER_LANE",
                                                                            "RIGHT_BOUNDARY", "CONE", "MOVABLE_SIGN",
                                                                            "LICENSE_PLATE"})]
        df = df.query("OBJECT_TYPE in @all_type")

        ts = np.arange(start_frame, start_frame + self.args.obs_len + self.args.pred_len)
        t_obs = ts[self.args.obs_len - 1]

        trajs, pad_flags = [], []
        pred_trajs, has_preds = [], []
        ego_lane_presence, agent_ids = [], []

        track_ids = np.unique(df["TRACK_ID"].values)
        # collect ego info
        df_ego = df[df["TRACK_ID"] == -1]
        ts_ego = np.array(df_ego["TIMESTAMP"].values).astype(np.int64)
        self.get_trajectory(df_ego, ts_ego, ts, t_obs, trajs, pad_flags, pred_trajs, has_preds, ego_lane_presence)
        agent_ids.append(-1)
        # collect obs info
        for idx in track_ids:
            df_agent = df[df["TRACK_ID"] == idx]
            if np.all(df_agent["OBJECT_TYPE"] == self.det_type["AV"]) and np.all(df_agent["TRACK_ID"] == -1):
                continue

            ts_agent = np.array(df_agent["TIMESTAMP"].values).astype(np.int64)
            if np.all(ts_agent > t_obs) or t_obs not in ts_agent:
                continue

            self.get_trajectory(df_agent, ts_agent, ts, t_obs,
                                trajs, pad_flags, pred_trajs, has_preds, ego_lane_presence)
            agent_ids.append(idx)

        # transform
        orig, rot = self.get_origin_rotation(trajs[0][:, :2])

        # predicted trajectory by relu-based method
        if len(pred_trajs) > 0:
            pred_trajs = (np.asarray(pred_trajs) - orig).dot(rot)
        else:
            pred_trajs = np.zeros((0, 50, 2))
        has_preds = np.asarray(has_preds).astype(np.int16)

        trajs = np.asarray(trajs, dtype=np.float_)
        trajs[:, :, :2] = (trajs[:, :, :2] - orig).dot(rot)
        trajs[:, :, 2:4] = trajs[:, :, 2:4].dot(rot)

        ts = (ts - ts[0]).astype(np.float32)
        trajs = trajs.astype(np.float32)
        pad_flags = np.array(pad_flags).astype(np.int16)
        ego_lane_presence = np.array(ego_lane_presence).astype(np.int16)
        agent_ids = np.array(agent_ids).astype(np.int64)

        return ts, trajs, pad_flags, orig, rot, pred_trajs, has_preds, ego_lane_presence, agent_ids

    def get_origin_rotation(self, agent_traj):
        orig = agent_traj[self.args.obs_len - 1]
        vec = orig - agent_traj[0]
        theta = np.arctan2(vec[1], vec[0])
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        return orig, rot

    def get_sorted_lane(self, df, orig):
        lanes = df[df["OBJECT_TYPE"] == self.det_type["CENTER_LANE"]]
        orig_shp = Point(orig)
        left_lane_ids, ego_lane_ids, right_lane_ids = dict(), dict(), dict()
        for i, idx in enumerate(lanes["TRACK_ID"].values):
            lane = lanes[lanes["TRACK_ID"] == idx]
            lane_cl = np.stack((lane["X"].values, lane["Y"].values), axis=1)
            dist_2_cl = LineString(lane_cl).distance(orig_shp)
            if np.all(lane["EGO_RELATION"] == self.ego_relation["LEFT"]):
                left_lane_ids[idx] = dist_2_cl
            elif np.all(lane["EGO_RELATION"] == self.ego_relation["RIGHT"]):
                right_lane_ids[idx] = dist_2_cl
            else:
                ego_lane_ids[idx] = dist_2_cl
        # left     ego      right
        # |   ...   |   ...   |
        # |   ...   |   ...   |
        # |   ...   |   ...   |
        left_lane_ids = {k: v for k, v in sorted(left_lane_ids.items(), key=lambda item: -item[1])}
        right_lane_ids = {k: v for k, v in sorted(right_lane_ids.items(), key=lambda item: item[1])}
        lane_ids = dict()
        for ids in [left_lane_ids, ego_lane_ids, right_lane_ids]:
            for k, v in ids.items():
                lane_ids[k] = v
        return lane_ids

    def get_lane_graph(self, df, orig, rot):
        # get left right edges at lane-level
        lane_ids = self.get_sorted_lane(df, orig)
        left_pairs, right_pairs = dict(), dict()
        left_turn, right_turn = np.zeros(len(lane_ids), np.float32), np.zeros(len(lane_ids), np.float32)
        for key in ["u", "v"]:
            left_pairs[key], right_pairs[key] = [], []
        left_pairs["u"].append(np.arange(1, len(lane_ids)))
        left_pairs["v"].append(np.arange(0, len(lane_ids) - 1))
        right_pairs["u"].append(np.arange(0, len(lane_ids) - 1))
        right_pairs["v"].append(np.arange(1, len(lane_ids)))
        left_turn[left_pairs["u"][0]] = 1
        right_turn[right_pairs["u"][0]] = 1

        lanes = df[df["OBJECT_TYPE"] == self.det_type["CENTER_LANE"]]

        # get ctrs and feats
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lanes["TRACK_ID"] == lane_id]
            lane_cl = np.stack((lane["X"].values, lane["Y"].values), axis=1)
            lane_cl[:, :2] = (lane_cl[:, :2] - orig).dot(rot)
            ctrs.append(np.asarray((lane_cl[1:] + lane_cl[:-1]) / 2.0, np.float_))
            feats.append(np.asarray((lane_cl[1:] - lane_cl[:-1]), np.float_))

            x = np.ones((len(ctrs[-1]), 2), np.float32)
            x[:, 0] *= left_turn[i]
            x[:, 1] *= right_turn[i]
            turn.append(x)

            control.append(np.zeros(len(ctrs[-1]), np.float32))
            intersect.append(np.zeros(len(ctrs[-1]), np.float32))

        # get node ids
        node_ids = []
        count = 0
        for i, ctr in enumerate(ctrs):
            node_ids.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count

        # get lane ids
        node_2_lane_ids = []  # node belongs to which lane
        for i, ids in enumerate(node_ids):
            node_2_lane_ids.append(i * np.ones(len(ids), np.int16))
        node_2_lane_ids = np.concatenate(node_2_lane_ids, 0)

        ctrs = np.concatenate(ctrs, axis=0)
        feats = np.concatenate(feats, axis=0)

        # get pre suc edges at node-level
        pre, suc = dict(), dict()
        for key in ["u", "v"]:
            pre[key], suc[key] = [], []
        for i in range(len(node_ids)):
            ids = node_ids[i]
            pre["u"] += ids[1:]
            pre["v"] += ids[:-1]
            suc["u"] += ids[:-1]
            suc["v"] += ids[1:]

        # get left right edges at node-level
        # using adjacency matrix to calculate dist among nodes which belong to two connected lanes
        # another way to solve the problem is using for-loop over left_pairs
        left, right = dict(), dict()
        num_lanes = len(lane_ids)

        dist = np.expand_dims(ctrs, axis=0) - np.expand_dims(ctrs, axis=1)
        dist = np.sqrt(np.sum(dist ** 2, axis=2))
        hi = np.arange(num_nodes).reshape(1, -1).repeat(num_nodes, axis=1).flatten()
        wi = np.arange(num_nodes).reshape(1, -1).repeat(num_nodes, axis=0).flatten()
        row_ids = np.arange(num_nodes)

        pairs = left_pairs
        if len(pairs) > 0:
            mat = np.zeros((num_lanes, num_lanes), dtype=np.int16)
            mat[pairs["u"][0], pairs["v"][0]] = 1  # adj matrix
            mat = mat > 0.5

            left_dist = dist.copy()
            mask = np.logical_not(mat[node_2_lane_ids[hi], node_2_lane_ids[wi]])
            left_dist[hi[mask], wi[mask]] = 1e6

            min_dist = np.min(left_dist, axis=1)
            min_ids = np.argmin(left_dist, axis=1)

            mask = min_dist < self.ls_cross_dist
            ui = row_ids[mask]
            vi = min_ids[mask]
            f1 = feats[ui]
            f2 = feats[vi]
            t1 = np.arctan2(f1[:, 1], f1[:, 0])
            t2 = np.arctan2(f2[:, 1], f2[:, 0])
            dt = np.abs(t1 - t2)
            m = dt > np.pi
            dt[m] = np.abs(dt[m] - 2 * np.pi)
            m = dt < 0.25 * np.pi

            ui = ui[m]
            vi = vi[m]

            left["u"] = ui.copy().astype(np.int16)
            left["v"] = vi.copy().astype(np.int16)
        else:
            left["u"] = np.zeros(0, np.int16)
            left["v"] = np.zeros(0, np.int16)

        pairs = right_pairs
        if len(pairs) > 0:
            mat = np.zeros((num_lanes, num_lanes), dtype=np.int16)
            mat[pairs["u"][0], pairs["v"][0]] = 1  # adj matrix
            mat = mat > 0.5

            right_dist = dist.copy()
            mask = np.logical_not(mat[node_2_lane_ids[hi], node_2_lane_ids[wi]])
            right_dist[hi[mask], wi[mask]] = 1e6

            min_dist = np.min(right_dist, axis=1)
            min_ids = np.argmin(right_dist, axis=1)

            mask = min_dist < self.ls_cross_dist
            ui = row_ids[mask]
            vi = min_ids[mask]
            f1 = feats[ui]
            f2 = feats[vi]
            t1 = np.arctan2(f1[:, 1], f1[:, 0])
            t2 = np.arctan2(f2[:, 1], f2[:, 0])
            dt = np.abs(t1 - t2)
            m = dt > np.pi
            dt[m] = np.abs(dt[m] - 2 * np.pi)
            m = dt < 0.25 * np.pi

            ui = ui[m]
            vi = vi[m]

            right["u"] = ui.copy().astype(np.int16)
            right["v"] = vi.copy().astype(np.int16)
        else:
            right["u"] = np.zeros(0, np.int16)
            right["v"] = np.zeros(0, np.int16)

        graph = dict()
        graph["num_nodes"] = num_nodes
        graph["ctrs"] = ctrs.astype(np.float32)
        graph["feats"] = feats.astype(np.float32)
        graph["turn"] = np.concatenate(turn, 0).astype(np.int16)
        graph["control"] = np.concatenate(control, 0).astype(np.int16)
        graph["intersect"] = np.concatenate(intersect, 0).astype(np.int16)
        graph["pre"] = [pre]
        graph["suc"] = [suc]
        graph["left"] = left
        graph["right"] = right
        graph["lane_ids"] = lane_ids

        for k1 in ["pre", "suc"]:
            for k2 in ["u", "v"]:
                graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int16)
        for key in ["pre", "suc"]:
            graph[key] += self.dilated_nbrs(graph[key][0], graph["num_nodes"], self.ls_num_scales)

        return graph

    def plot_trajs(self, ax, trajs, pad_flags, orig, rot, vis_map=True):
        if not vis_map:
            rot = np.eye(2)
            orig = np.zeros(2)

        for i, traj in enumerate(trajs):
            zorder = 10
            if i == 0:  # agent
                clr = "r"
                zorder = 20
            else:
                clr = "orange"

            traj = traj.dot(rot.T) + orig
            ax.plot(traj[:, 0], traj[:, 1], marker=".", alpha=0.5, color=clr, zorder=zorder)
            ax.text(traj[self.args.obs_len, 0], traj[self.args.obs_len, 1], str(i))
            ax.scatter(traj[:, 0], traj[:, 1], s=list((1 - pad_flags[i]) * 50 + 1), color="b")

    def plot_lane_graph(self, ax, df, graph, orig, rot, vis_map=True):
        x_min = orig[0] - 100
        x_max = orig[0] + 100
        y_min = orig[1] - 100
        y_max = orig[1] + 100
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if vis_map:
            lanes = df[df["OBJECT_TYPE"] == self.det_type["CENTER_LANE"]]
            lane_ids = lanes["TRACK_ID"].unique()
            for lane_id in lane_ids:
                lane = lanes[lanes["TRACK_ID"] == lane_id]
                lane_cl = np.stack((lane["X"].values, lane["Y"].values), axis=1)
                pt = lane_cl[0]
                vec = lane_cl[1] - lane_cl[0]
                # ax.scatter(lane_cl[:, 0], lane_cl[:, 1], color="black", s=3, alpha=0.5)
                ax.plot(lane_cl[:, 0], lane_cl[:, 1], color="grey", alpha=0.5)
                ax.arrow(pt[0], pt[1], vec[0], vec[1], alpha=0.5, color="grey", width=0.1, zorder=1)
                m_pt = lane_cl[int(lane_cl.shape[0] / 2)]
                ax.text(m_pt[0], m_pt[1], "{:d}".format(int(lane_id)), color="b")
        else:
            rot = np.eye(2)
            orig = np.zeros(2)

        ctrs = graph["ctrs"]
        ctrs[:, :2] = ctrs[:, :2].dot(rot.T) + orig
        ax.scatter(ctrs[:, 0], ctrs[:, 1], color="b", s=2, alpha=0.5)

        feats = graph["feats"]
        feats[:, :2] = feats[:, :2].dot(rot.T)
        for j in range(feats.shape[0]):
            vec = feats[j]
            pt0 = ctrs[j] - vec / 2
            pt1 = ctrs[j] + vec / 2
            ax.arrow(pt0[0], pt0[1], (pt1 - pt0)[0], (pt1 - pt0)[1], edgecolor=None, color="deepskyblue", alpha=0.3)

        left_u = graph["left"]["u"]
        left_v = graph["left"]["v"]
        for u, v in zip(left_u, left_v):
            x = ctrs[u]
            dx = ctrs[v] - ctrs[u]
            ax.arrow(x[0], x[1], dx[0], dx[1], edgecolor=None, color="green", alpha=0.3)

        right_u = graph["right"]["u"]
        right_v = graph["right"]["v"]
        for u, v in zip(right_u, right_v):
            x = ctrs[u]
            dx = ctrs[v] - ctrs[u]
            ax.arrow(x[0], x[1], dx[0], dx[1], edgecolor=None, color="green", alpha=0.3)

    @staticmethod
    def dilated_nbrs(nbr, num_nodes, num_scales):
        data = np.ones(len(nbr["u"]), np.bool_)
        csr = sparse.csr_matrix((data, (nbr["u"], nbr["v"])), shape=(num_nodes, num_nodes))

        mat = csr
        nbrs = []
        for _ in range(1, num_scales):
            mat = mat * mat

            nbr = dict()
            coo = mat.tocoo()
            nbr["u"] = coo.row.astype(np.int16)
            nbr["v"] = coo.col.astype(np.int16)
            nbrs.append(nbr)
        return nbrs

    @staticmethod
    def padding_traj_nn(traj):
        n = len(traj)

        # forward
        buff = None
        for i in range(n):
            if np.all(buff == None) and np.all(traj[i] != None):
                buff = traj[i]
            if np.all(buff != None) and np.all(traj[i] == None):
                traj[i] = buff
            if np.all(buff != None) and np.all(traj[i] != None):
                buff = traj[i]

        # backward
        buff = None
        for i in reversed(range(n)):
            if np.all(buff == None) and np.all(traj[i] != None):
                buff = traj[i]
            if np.all(buff != None) and np.all(traj[i] == None):
                traj[i] = buff
            if np.all(buff != None) and np.all(traj[i] != None):
                buff = traj[i]

        return traj


def get_args():
    parser = argparse.ArgumentParser(description="Echo PlusAI custom pkl messages")
    parser.add_argument("-f", "--file_dir",
                        type=str,
                        default="./data/processed/",
                        help="Path to the pkl file for extracting data")
    parser.add_argument("-s", "--save_dir",
                        default="./data/features/",
                        type=str,
                        help="Path where the computed features are saved")
    parser.add_argument("-m", "--mode",
                        required=True,
                        type=str,
                        help="train/val/test")
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
    parser.add_argument("--ego_lead",
                        default=False,
                        action="store_true",
                        help="Build dataset for ego lead")

    return parser.parse_args()


def save_as_argo(dataset, files, start_idx, batch_size, args):
    loop = tqdm(range(batch_size), total=batch_size, leave=True)
    for i in loop:
        if i + start_idx < len(files):
            file = files[start_idx + i]
            scenario = pd.read_pickle(os.path.join(args.file_dir, file))
            file_name = file.split('.')[0][:-5]
            data, headers = dataset.process(file_name, scenario)
            if np.all(data[0][-1] != 2) and args.ego_lead:
                continue
            if os.path.isabs(args.save_dir):
                save_dir = os.path.join(args.save_dir, args.mode, file_name + "_argo" + ".pkl")
            else:
                save_dir = os.path.join(os.getcwd(), args.save_dir, args.mode, file_name + "_argo" + ".pkl")
            if not os.path.exists(save_dir) and not args.debug:
                data_df = pd.DataFrame(data, columns=headers)
                data_df.to_pickle(save_dir)


def main():
    start = time.time()

    args = get_args()
    dataset = PreProcess(args)

    files = os.listdir(args.file_dir)
    num_files = len(files)
    print("num of files: ", num_files)

    n_proc = multiprocessing.cpu_count() - 2 if not args.debug else 1
    batch_size = np.max([int(np.ceil(num_files / float(n_proc))), 1])
    print('n_proc: {}, batch_size: {}'.format(n_proc, batch_size))

    Parallel(n_jobs=n_proc)(delayed(save_as_argo)(dataset, files, i, batch_size, args)
                            for i in range(0, num_files, batch_size))
    print("Preprocess for {} set completed in {} minutes".format(args.file_dir, (time.time() - start) / 60.0))


if __name__ == "__main__":
    main()
