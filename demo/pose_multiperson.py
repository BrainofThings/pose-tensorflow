import click
import yaml
import os
import sys
import time
import cv2
import bisect

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
# HACK: Remove learning repo since it has its own dataset module which conflicts with this repo
learning_repo_path = '/home/brainoft/learning/offline_learning'
if learning_repo_path in sys.path:
    sys.path.remove(learning_repo_path)

from scipy.misc import imread
from config import load_config
from dataset.factory import create as create_dataset
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

from multiperson.detections import extract_detections
from multiperson.predict import SpatialModel, eval_graph, get_person_conf_multicut
from multiperson.visualize import PersonDraw, visualize_detections

import matplotlib.pyplot as plt

OPEN_POSE_CONFIG = {
    'debug': True,
    'ignore_areas': [],
}


@click.group()
def cli():
    pass


def get_host_configs(hostname):
    config_file = '/home/brainoft/learning/offline_learning/poc_data/default/configuration.yaml'
    poc_config_file = '/home/brainoft/learning/offline_learning/poc_data/%s/configuration.yaml' % hostname
    if os.path.isfile(poc_config_file):
        config_file = poc_config_file
    with open(config_file) as config_yaml:
            return yaml.load(config_yaml.read())


class ImageIndex(object):

    def __init__(self, rootdir):
        self._index = ImageIndex._create_image_index(rootdir)

    def cameras(self):
        """
        Return the list of cameras.
        """
        return self._index.keys()

    def image_filename(self, camera, t_sec, with_timestamp=False):
        """
        Return the image filename that is just before the time t_sec.
        """
        chosen_start_time_ms = -1
        for start_time_ms in self._index[camera]:
            if chosen_start_time_ms < start_time_ms <= t_sec * 1000:
                chosen_start_time_ms = start_time_ms
        if chosen_start_time_ms == -1:
            if not with_timestamp:
                return None
            return None, None
        timestamps = self._index[camera][chosen_start_time_ms]['timestamps']
        index = bisect.bisect_left(timestamps, t_sec * 1000)
        while index >= len(timestamps) or index > 0 and timestamps[index] > t_sec * 1000:
            index -= 1

        img_path = os.path.join(self._index[camera][chosen_start_time_ms]['dir_path'],
                                '{}.{}'.format(timestamps[index],
                                               self._index[camera][chosen_start_time_ms]['extension']))
        if not with_timestamp:
            return img_path
        return img_path, timestamps[index] / 1000.0

    @staticmethod
    def _create_image_index(image_rootdir):
        image_index = {}
        for dir_name in [name for name in os.listdir(image_rootdir)
                         if os.path.isdir(os.path.join(image_rootdir, name))]:
            split_dir_name = dir_name.split('_')
            camera_name = '_'.join(split_dir_name[:-1])
            print("Creating index for {}".format(camera_name))

            image_filenames = [name for name in os.listdir(os.path.join(image_rootdir, dir_name))
                               if os.path.isfile(os.path.join(image_rootdir, dir_name, name))
                               and (name.endswith('.jpg') or name.endswith('.png'))]
            print("Indexing {}".format(len(image_filenames)))
            if not image_filenames:
                # No images in this directory
                print("Did not find any images for camera {} in {}".format(camera_name, dir_name))
                continue
            image_timestamps = []
            for name in image_filenames:
                try:
                    image_timestamps.append(int(name.split('.')[0]))
                except:
                    print('Problem with filename: {}'.format(name))
            image_timestamps = sorted(image_timestamps)
            start_time = image_timestamps[0]
            print("Indexed {} images for {} from {}".format(len(image_timestamps), camera_name, dir_name))

            if camera_name not in image_index:
                image_index[camera_name] = {}
            image_index[camera_name][start_time] = {
                'dir_path': os.path.join(image_rootdir, dir_name),
                'timestamps': image_timestamps,
                'extension': image_filenames[0][-3:]
            }
        return image_index


def read_image(image_path, ignore_areas=[]):
    try:
        # NOTE: This has to be read in RGB
        # TODO: Figure out if cv2.imread followed by cv2.COLOR_BGR2RGB works
        #frame = cv2.imread(image_path)
        #First convert to BGR if its gray scale and then convert to RGB 
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = imread(image_path, mode='RGB')
        for area in ignore_areas:
            if len(frame.shape) == 3:
                frame[area['y1']: area['y2'], area['x1']:area['x2']] = (0, 0, 0)
            else:
                frame[area['y1']: area['y2'], area['x1']:area['x2']] = 0
        return frame
    except:
        print('Cannont read the file %s' % image_path)


def morph(img):
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    mask = mask / 255
    return img * mask


@click.command()
@click.argument('hostname', type=str)
@click.argument('img_dir', type=click.Path(file_okay=False))
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.option('--begin_time', '-b', type=float, default=None, help="start time in epoch seconds")
@click.option('--end_time', '-e', type=float, default=None, help="stop time in epoch seconds")
@click.option('--step_size', '-s', type=float, default=0.5, help="step size in seconds")
@click.option("--camera_list", "-c", type=str, default=None,
              help="Comma separated list of cameras for which heatmap should be generated")
def compute_pose_plot(hostname, img_dir, output_dir, begin_time, end_time, step_size, camera_list):
    host_configs = get_host_configs(hostname)
    cameras = camera_list.split(",")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create image index
    start = time.time()
    image_index = ImageIndex(img_dir)
    print("Image index created in {} secs".format(time.time() - start))

    # load config
    start = time.time()
    cfg = load_config("demo/pose_cfg_multi.yaml")
    dataset = create_dataset(cfg)
    print("Configs loaded in {} secs".format(time.time() - start))

    # # load model
    start = time.time()
    sm = SpatialModel(cfg)
    sm.load()
    draw_multi = PersonDraw()
    # Load and setup CNN part detector
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)
    print("Model loaded in {} secs".format(time.time() - start))

    # process images
    for camera in cameras:
        camera_output_dir = os.path.join(output_dir, camera) if len(cameras) > 1 else output_dir
        if not os.path.exists(camera_output_dir):
            os.makedirs(camera_output_dir)
        camera_params = host_configs["camera_bounds"]
        if camera in camera_params and "open_pose" in camera_params[camera]:
            config = camera_params[camera]["open_pose"]
            print("Using config from configuration file:\n{}".format(config))
        else:
            config = OPEN_POSE_CONFIG
        prev_img = None
        for t_sec in np.arange(begin_time, end_time + step_size, step_size):
            print("Processing {}".format(t_sec))
            start = time.time()
            img_path = image_index.image_filename(camera, t_sec)
            img_rgb = read_image(img_path, config['ignore_areas'])
            #if len(img.shape) == 2:
            #    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            #else:
            #    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            image_batch = data_to_input(img_rgb)

            # Compute prediction with the CNN
            outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
            scmap, locref, pairwise_diff = predict.extract_cnn_output(outputs_np, cfg, dataset.pairwise_stats)

            detections = extract_detections(cfg, scmap, locref, pairwise_diff)
            unLab, pos_array, unary_array, pwidx_array, pw_array = eval_graph(sm, detections)
            person_conf_multi = get_person_conf_multicut(sm, unLab, unary_array, pos_array)
           
            background = np.zeros_like(img_rgb) 
            pose_image = draw_multi.draw_pose(background, dataset, person_conf_multi)

            output_file = os.path.join(camera_output_dir, '{}.jpg'.format(int(np.around(t_sec * 1000))))
            cv2.imwrite(output_file, pose_image)
            print("Finished processing {} in {} secs".format(t_sec, time.time() - start))


if __name__ == '__main__':
    cli.add_command(compute_pose_plot)
    cli()

