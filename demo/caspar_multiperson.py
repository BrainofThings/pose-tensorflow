import cv2
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from scipy.misc import imread, imsave

from config import load_config
from dataset.factory import create as create_dataset
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

from multiperson.detections import extract_detections
from multiperson.predict import SpatialModel, eval_graph, get_person_conf_multicut
from multiperson.visualize import PersonDraw, visualize_detections

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


cfg = load_config("demo/pose_cfg_multi.yaml")

dataset = create_dataset(cfg)

sm = SpatialModel(cfg)
sm.load()

draw_multi = PersonDraw()

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read image from file
file_name = "demo/image_multi.png"
image = imread(file_name, mode='RGB')
image_batch = data_to_input(image)

# Compute prediction with the CNN
outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
scmap, locref, pairwise_diff = predict.extract_cnn_output(outputs_np, cfg, dataset.pairwise_stats)

detections = extract_detections(cfg, scmap, locref, pairwise_diff)
unLab, pos_array, unary_array, pwidx_array, pw_array = eval_graph(sm, detections)
person_conf_multi = get_person_conf_multicut(sm, unLab, unary_array, pos_array)

pose_image = draw_multi.draw_pose(image, dataset, person_conf_multi)
cv2.imwrite('/home/brainoft/Desktop/caspar_pose.jpg', pose_image)
