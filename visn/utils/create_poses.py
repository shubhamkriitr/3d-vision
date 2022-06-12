# Extract ground truth poses from original ScanNet data and add them to
# the UprightNet ScanNet data

from pathlib import Path

import sys
sys.path.append('/home/timo/git/3dvision/ScanNet/SensReader/python')
from SensorData import SensorData

import os.path
from os import path

from os import listdir
from os.path import isfile, join

from distutils.dir_util import copy_tree

normal_list_path = "/media/timo/LaCie1/scannet/test_scannet_normal_list.txt"  # list has to contain original ScanNet data instead of UprightNet data (http://kaldir.vc.in.tum.de/scannet/download-scannet.py)
target_path = "/media/timo/LaCie1/scannet/our_data/" # folder where ground truth poses are saved to

with open(normal_list_path) as f:
    normal_list = f.read().splitlines()

#example_string = "/media/timo/LaCie1/scannet/data/scene0144_01/normal_pair/0.png"
for example_string in normal_list:
    # create pose files[]
    output_data_folder = example_string.split("/normal_pair/", 1)[0]
    image_data_folder = output_data_folder + "/normal_pair"
    scene_folder = output_data_folder.split("/data/", 1)[1]

    if(not path.exists(target_path + scene_folder)):
        copy_tree(output_data_folder, target_path + scene_folder)

    output_data_folder = target_path + scene_folder
    image_data_folder = output_data_folder + "/normal_pair"

    input_folder = example_string.split("/data/", 1)[0] + "/scans/" + scene_folder + "/" + scene_folder + ".sens"

    if(path.exists(output_data_folder) and not path.exists(output_data_folder+"/pose") and path.exists(input_folder)):
        sd = SensorData(input_folder)

        frames = []
        for f in listdir(image_data_folder):
            if isfile(join(image_data_folder, f)):
                frames.append(int(f.split(".", 1)[0]))

        sd.export_specific_poses(os.path.join(output_data_folder+"/pose"), frames)
    else:
       print(output_data_folder, " does not exist.")