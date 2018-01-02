import cv2
import math
import numpy as np

import scipy.spatial

import matplotlib.pyplot as plt

import munkres

from util.visualize import check_point, _npcircle
from util import visualize


min_match_dist = 200
marker_size = 5

draw_conf_min_count = 3

# Colors
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
CYAN = (255, 0, 0)
PURPLE = (128, 0, 128)
VIOLET = (238, 130, 238)
MEDIUM_TURQOISE = (204, 209, 72)
MEDIUM_BLUE = (180, 0, 0)
SKY_BLUE = (235, 206, 135)
ORANGE = (0, 165, 255)
GOLD = (0, 215, 255)
ORANGE_RED = (0, 69, 255)
LIGHT_BLUE = (230, 216, 173)
LIGHT_GREEN = (144, 238, 144)
LIGHT_YELLOW = (224, 255, 255)
LIME_GREEN = (50, 205, 50)
YELLOW_GREEN = (50, 205, 154)
DARK_GREEN = (0, 100, 0)

JOINT_COLORS = {
    (0,1): BLUE,
    (0,2): LIME_GREEN,
    (1,3): GOLD,
    (2,4): CYAN,
    (5,7): ORANGE_RED,
    (5,17): MEDIUM_TURQOISE,
    (6,8): ORANGE,
    (6,17): VIOLET,
    (7,9): ORANGE,
    (8,10): ORANGE_RED,
    (11,13): VIOLET,
    (11,17): YELLOW,
    (12,14): MEDIUM_TURQOISE,
    (12,17): GREEN,
    (13,15): MEDIUM_TURQOISE,
    (14,16): VIOLET,
}

FACE_JOINTS = [(0,1), (0,2), (1,3), (2,4)]


def get_ref_points(person_conf):
    avg_conf = np.sum(person_conf, axis=1) / person_conf.shape[1]

    # last points is tip of the head -> use it as reference
    ref_points = person_conf[:, -1, :]

    # use average of other points if head tip is missing
    emptyidx = (np.sum(ref_points, axis=1) == 0)
    ref_points[emptyidx, :] = avg_conf[emptyidx, :]

    return ref_points


class PersonDraw:
    def __init__(self):
        self.mk = munkres.Munkres()

        self.prev_person_conf = np.zeros([0, 1])
        self.prev_color_assignment = None

        # generated colors from http://tools.medialab.sciences-po.fr/iwanthue/
        track_colors_str = ["#F5591E",
                            "#3870FB",
                            "#FE5DB0",
                            "#B4A691",
                            "#43053F",
                            "#3475B1",
                            "#642612",
                            "#B3B43D",
                            "#DD9BFE",
                            "#28948D",
                            "#E99D53",
                            "#012B46",
                            "#9D2DA3",
                            "#04220A",
                            "#62CB22",
                            "#EE8F91",
                            "#D71638",
                            "#00613A",
                            "#318918",
                            "#B770FF",
                            "#82C091",
                            "#6C1333",
                            "#973405",
                            "#B19CB2",
                            "#F6267B",
                            "#284489",
                            "#97BF17",
                            "#3B899C",
                            "#931813",
                            "#FA76B6"]

        self.track_colors = [(int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)) for s in track_colors_str]

    def draw(self, visim, dataset, person_conf):
        minx = 2 * marker_size
        miny = 2 * marker_size
        maxx = visim.shape[1] - 2 * marker_size
        maxy = visim.shape[0] - 2 * marker_size

        num_people = person_conf.shape[0]
        color_assignment = dict()

        # MA: assign same color to matching body configurations
        if self.prev_person_conf.shape[0] > 0 and person_conf.shape[0] > 0:
            ref_points = get_ref_points(person_conf)
            prev_ref_points = get_ref_points(self.prev_person_conf)

            # MA: this munkres implementation assumes that num(rows) >= num(columns)
            if person_conf.shape[0] <= self.prev_person_conf.shape[0]:
                cost_matrix = scipy.spatial.distance.cdist(ref_points, prev_ref_points)
            else:
                cost_matrix = scipy.spatial.distance.cdist(prev_ref_points, ref_points)

            assert (cost_matrix.shape[0] <= cost_matrix.shape[1])

            conf_assign = self.mk.compute(cost_matrix)

            if person_conf.shape[0] > self.prev_person_conf.shape[0]:
                conf_assign = [(idx2, idx1) for idx1, idx2 in conf_assign]
                cost_matrix = cost_matrix.T

            for pidx1, pidx2 in conf_assign:
                if cost_matrix[pidx1][pidx2] < min_match_dist:
                    color_assignment[pidx1] = self.prev_color_assignment[pidx2]

        print("#tracked objects:", len(color_assignment))

        free_coloridx = sorted(list(set(range(len(self.track_colors))).difference(set(color_assignment.values()))),
                               reverse=True)

        for pidx in range(num_people):
            # color_idx = pidx % len(self.track_colors)
            if pidx in color_assignment:
                color_idx = color_assignment[pidx]
            else:
                if len(free_coloridx) > 0:
                    color_idx = free_coloridx[-1]
                    free_coloridx = free_coloridx[:-1]
                else:
                    color_idx = np.random.randint(len(self.track_colors))

                color_assignment[pidx] = color_idx

            assert (color_idx < len(self.track_colors))

            if np.sum(person_conf[pidx, :, 0] > 0) < draw_conf_min_count:
                continue

            for kidx1, kidx2 in dataset.get_pose_segments():
                p1 = (int(math.floor(person_conf[pidx, kidx1, 0])), int(math.floor(person_conf[pidx, kidx1, 1])))
                p2 = (int(math.floor(person_conf[pidx, kidx2, 0])), int(math.floor(person_conf[pidx, kidx2, 1])))

                if check_point(p1[0], p1[1], minx, miny, maxx, maxy) and check_point(p2[0], p2[1], minx, miny, maxx,
                                                                                     maxy):
                    color = np.array(self.track_colors[color_idx][::-1], dtype=np.float64) / 255.0
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], marker='o', linestyle='solid', linewidth=2.0, color=color)


        self.prev_person_conf = person_conf
        self.prev_color_assignment = color_assignment

    def draw_pose(self, visim, dataset, person_conf):
        visim = cv2.cvtColor(visim, cv2.COLOR_BGR2RGB)
        minx = 2 * marker_size
        miny = 2 * marker_size
        maxx = visim.shape[1] - 2 * marker_size
        maxy = visim.shape[0] - 2 * marker_size

        num_people = person_conf.shape[0]
        for pidx in range(num_people):
            if np.sum(person_conf[pidx, :, 0] > 0) < draw_conf_min_count:
                continue

            joint_coords = {}
            for kidx1, kidx2 in dataset.get_pose_segments():
                #print("Joint: {} {}, Color: {}".format(kidx1, kidx2, JOINT_COLORS[(kidx1, kidx2)]))
                p1 = (int(math.floor(person_conf[pidx, kidx1, 0])), int(math.floor(person_conf[pidx, kidx1, 1])))
                p2 = (int(math.floor(person_conf[pidx, kidx2, 0])), int(math.floor(person_conf[pidx, kidx2, 1])))
                if check_point(p1[0], p1[1], minx, miny, maxx, maxy) and check_point(p2[0], p2[1], minx, miny, maxx,
                                                                                     maxy):
                    joint_coords[kidx1] = p1
                    joint_coords[kidx2] = p2
                    if (kidx1, kidx2) in FACE_JOINTS:
                        continue
                    cv2.line(visim, p1, p2, JOINT_COLORS[(kidx1, kidx2)], 3)
                    cv2.circle(visim, p1, 3, WHITE, thickness=-1)
                    cv2.circle(visim, p2, 3, WHITE, thickness=-1)
            if 5 in joint_coords and 6 in joint_coords:
                x, y = tuple(map(lambda l, r: int((l + r)/2), joint_coords[5], joint_coords[6]))
                neck = (x, y-3)
                cv2.line(visim, joint_coords[5], neck, JOINT_COLORS[(5,17)], 3)
                cv2.line(visim, joint_coords[6], neck, JOINT_COLORS[(6,17)], 3)
                cv2.circle(visim, joint_coords[5], 3, WHITE, thickness=-1)
                cv2.circle(visim, joint_coords[6], 3, WHITE, thickness=-1)
                if 11 in joint_coords:
                    cv2.line(visim, joint_coords[11], neck, JOINT_COLORS[(11,17)], 3)
                    cv2.circle(visim, joint_coords[11], 3, WHITE, thickness=-1)
                if 12 in joint_coords:
                    cv2.line(visim, joint_coords[12], neck, JOINT_COLORS[(12,17)], 3)
                    cv2.circle(visim, joint_coords[12], 3, WHITE, thickness=-1)
                cv2.circle(visim, neck, 3, WHITE, thickness=-1)
            if 4 in joint_coords:
                cv2.circle(visim, joint_coords[4], 4, MAGENTA, thickness=-1)
            if 3 in joint_coords:
                cv2.circle(visim, joint_coords[3], 4, PURPLE, thickness=-1)
            if 2 in joint_coords:
                cv2.circle(visim, joint_coords[2], 4, GOLD, thickness=-1)
            if 1 in joint_coords:
                cv2.circle(visim, joint_coords[1], 4, ORANGE, thickness=-1)
            if 0 in joint_coords:
                cv2.circle(visim, joint_coords[0], 4, ORANGE_RED, thickness=-1)
        return visim



keypoint_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0),
                   (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0),
                   (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

def visualize_detections(cfg, img, detections):
    vis_scale = 1.0
    marker_size = 4

    minx = 2 * marker_size
    miny = 2 * marker_size
    maxx = img.shape[1] - 2 * marker_size
    maxy = img.shape[0] - 2 * marker_size

    unPos = detections.coord
    joints_to_visualise = range(cfg.num_joints)
    visim_dets = img.copy()
    for pidx in joints_to_visualise:
        for didx in range(unPos[pidx].shape[0]):
            cur_x = unPos[pidx][didx, 0] * vis_scale
            cur_y = unPos[pidx][didx, 1] * vis_scale

            # / cfg.global_scale

            if check_point(cur_x, cur_y, minx, miny, maxx, maxy):
                _npcircle(visim_dets,
                          cur_x, cur_y,
                          marker_size,
                          keypoint_colors[pidx])
    return visim_dets
