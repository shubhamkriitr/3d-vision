from pathlib import Path
import sys
import os.path
from os import path

from os import listdir
from os.path import isfile, isdir, join
import shutil
import numpy as np
from distutils.dir_util import copy_tree

visn_data_path = "/media/timo/LaCie1/scannet/visn"

# get all scenes with data
scene_names = []
for f in listdir(visn_data_path):
    if isdir(join(visn_data_path, f)):
        scene_names.append(f)

# calculate error
angle_errors = []
scene_grav_errors = []
for scene in scene_names:
    scene_path = visn_data_path + "/" + scene
    grav_error_avg = 0
    grav_count = 0
    for f in listdir(scene_path+"/roll_pitch_gt"):
        if isfile(scene_path + "/roll_pitch_gt/" + f):
            gt_file = open(scene_path + "/roll_pitch_gt/" + f, "r")
            pred_file = open(scene_path + "/roll_pitch_pred/" + f, "r")
            grav_gt_file = open(scene_path + "/gravity_gt/" + f, "r")
            grav_pred_file = open(scene_path + "/gravity_pred/" + f, "r")

            roll_pitch_gt = gt_file.readline()
            roll_pitch_pred = pred_file.readline()
            grav_gt = grav_gt_file.readline()
            grav_pred = grav_pred_file.readline()
            roll_gt = float(roll_pitch_gt.split(" ", 1)[0])
            pitch_gt = float(roll_pitch_gt.split(" ", 1)[1])
            roll_pred = float(roll_pitch_pred.split(" ", 1)[0])
            pitch_pred = float(roll_pitch_pred.split(" ", 1)[1])
            image_path = scene_path + "/images/" + f.split(".", 1)[0]+ ".png"
            roll_error = abs(roll_gt-roll_pred)
            pitch_error = abs(pitch_gt-pitch_pred)
            grav_error_vec = [ float(grav_gt.split(" ", 3)[i]) - float(grav_pred.split(" ", 3)[i]) for i in range(3) ]
            grav_error = np.sqrt(grav_error_vec[0]**2 + grav_error_vec[1]**2 + grav_error_vec[2]**2)
            angle_errors.append([image_path, roll_error, pitch_error, grav_error])

            grav_error_avg += grav_error
            grav_count += 1

    scene_grav_errors.append([scene, grav_error_avg/grav_count])
    print(scene)

angle_errors = sorted(angle_errors,key=lambda x: (x[3]))
scene_grav_errors = sorted(scene_grav_errors,key=lambda x: (x[1]))

error_file = open(visn_data_path + "/roll_pitch_error.txt", "w")
for i in range(len(angle_errors)):
    write_str = ""
    for j in range(len(angle_errors[i][:])):
        write_str += str(angle_errors[i][j])
        if j < 3:
            write_str += ","
    error_file.write(write_str + "\n")

error_file.close()


scene_error_file = open(visn_data_path + "/scene_error.txt", "w")
for i in range(len(scene_grav_errors)):
    write_str = str(scene_grav_errors[i][0]) + "," + str(scene_grav_errors[i][1])
    scene_error_file.write(write_str + "\n")
scene_error_file.close()