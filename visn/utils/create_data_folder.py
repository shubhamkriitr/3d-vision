from pathlib import Path
import sys
import os.path
from os import path

from os import listdir
from os.path import isfile, isdir, join, exists
import shutil

from distutils.dir_util import copy_tree

our_data_path = "/media/timo/LaCie1/scannet/our_data"
visn_data_path = "/media/timo/LaCie1/scannet/visn"

# get all scenes with data
scene_names = []
for f in listdir(our_data_path):
    if isdir(join(our_data_path, f)):
        scene_names.append(f)

# create the folder structure
count = 0
for scene in scene_names:
    if exists(os.path.join(visn_data_path, scene)):
        print(scene + " already exists.")
        count += 1
        continue

    # create folders
    scene_path = visn_data_path + "/" + scene
    Path(scene_path).mkdir(parents=True, exist_ok=True)  # scene1234_01
    Path(scene_path + "/calibration").mkdir(parents=True, exist_ok=True)
    Path(scene_path + "/images").mkdir(parents=True, exist_ok=True)
    Path(scene_path + "/relative_pose").mkdir(parents=True, exist_ok=True)
    Path(scene_path + "/roll_pitch_gt").mkdir(parents=True, exist_ok=True)
    Path(scene_path + "/roll_pitch_pred").mkdir(parents=True, exist_ok=True)
    Path(scene_path + "/gravity_gt").mkdir(parents=True, exist_ok=True)
    Path(scene_path + "/gravity_pred").mkdir(parents=True, exist_ok=True)

    # create camera intrinsic K
    f = open(our_data_path+"/"+scene+"/intrinsic/intrinsic_resized.txt", "r")
    k_file = open(scene_path + "/calibration/K.txt", "w")
    for i in range(3):
        line = f.readline()
        k_line = line.strip()
        if i < 2:
            k_line += "\n"
        k_file.write(k_line)
    k_file.close()

    # create image size file
    file = open(scene_path + "/calibration/image_size.txt", "w")
    file.write("640\n480")
    file.close()

    # get file numbers
    image_files = os.listdir(os.path.join(our_data_path, scene, "rgb"))
    image_numbers = []
    for image_name in image_files:
        image_nr = int(image_name.split(".", 1)[0])
        image_numbers.append(image_nr)
    image_numbers.sort()
    
    # copy files to folders
    group_file = open(scene_path + "/groups.txt", "w")

    last_image_nr = -1
    start = True
    for image_nr in image_numbers:
        if last_image_nr != -1 and start:
            group_file.write(f"{(last_image_nr):04}" + " " + f"{(image_nr):04}")
            start = False
        elif last_image_nr != -1:
            group_file.write("\n" + f"{(last_image_nr):04}" + " " + f"{(image_nr):04}")
        last_image_nr = image_nr

        shutil.copy(our_data_path + "/" + scene + "/rgb/" + str(image_nr) + ".png", scene_path + "/images/" + f"{(image_nr):04}" + ".png")
        
        pose_file_src = open(our_data_path + "/" + scene + "/pose/" + str(image_nr) + ".txt", "r")
        pose_file_dest = open(scene_path + "/relative_pose/" + f"{(image_nr):04}" + ".txt", "w")
        lines = pose_file_src.readlines()
        pose_file_dest.write(lines[0])
        pose_file_dest.write(lines[1])
        pose_file_dest.write(lines[2].strip())
        pose_file_src.close()
        pose_file_dest.close()

        shutil.copy(our_data_path + "/" + scene + "/pose_gt/" + str(image_nr) + ".txt", scene_path + "/roll_pitch_gt/" + f"{(image_nr):04}" + ".txt")
        shutil.copy(our_data_path + "/" + scene + "/pose_pred/" + str(image_nr) + ".txt", scene_path + "/roll_pitch_pred/" + f"{(image_nr):04}" + ".txt")
        shutil.copy(our_data_path + "/" + scene + "/gravity_gt/" + str(image_nr) + ".txt", scene_path + "/gravity_gt/" + f"{(image_nr):04}" + ".txt")
        shutil.copy(our_data_path + "/" + scene + "/gravity_pred/" + str(image_nr) + ".txt", scene_path + "/gravity_pred/" + f"{(image_nr):04}" + ".txt")

    group_file.close()

    count += 1
    print(str(count/len(scene_names)*100) + "%")