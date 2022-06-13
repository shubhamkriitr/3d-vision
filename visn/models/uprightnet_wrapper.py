from __future__ import division
import os
from pathlib import Path
import time
import torch
import numpy as np
from torch.autograd import Variable
import UprightNet.models.networks
from UprightNet.options.test_options import TestOptions 
import sys
from UprightNet.data.data_loader import *
from UprightNet.models.models import create_model
import random
from tensorboardX import SummaryWriter

class UprightNetWrapper(object):
    """ Wrapper for UprightNet which allows for computing the upright vector
        used for the 3-point solver in pose estimation.
    """
    def __init__(self, datapath):
        opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
        self.model = create_model(opt, _isTrain=False)
        self.model.switch_to_eval()

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        eval_list_path = datapath
        eval_num_threads = 1
        EVAL_BATCH_SIZE = 8
        test_data_loader = CreateScanNetDataLoader(opt, eval_list_path, 
                                                        False, EVAL_BATCH_SIZE, 
                                                        eval_num_threads)
        self.test_dataset = test_data_loader.load_data()
        test_data_size = len(test_data_loader)
        print('========================= ScanNet eval #images = %d ========='%test_data_size)


    def get_gravity_vector(self):

        pred_roll_list = []
        pred_pitch_list = []
        gt_roll_list = []
        gt_pitch_list = []

        for i, data in enumerate(self.test_dataset):
            stacked_img = data[0]
            targets = data[1]

            pred_roll, pred_pitch, gt_roll, gt_pitch = self.model.test_roll_pitch(stacked_img, targets)

            pred_roll_list = pred_roll_list + pred_roll
            pred_pitch_list = pred_pitch_list + pred_pitch
            gt_roll_list = gt_roll_list + gt_roll
            gt_pitch_list = gt_pitch_list + gt_pitch

        return pred_roll_list, pred_pitch_list, gt_roll_list, gt_pitch_list


# todo: remove
datapath = os.path.join(Path.home(), 'data/test_scannet_normal_list.txt')

wrapper = UprightNetWrapper(datapath)
pred_roll_list, pred_pitch_list, gt_roll_list, gt_pitch_list = wrapper.get_gravity_vector()