from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import torchvision.transforms.functional as TF
import torchvision
import torch
import logging
import PIL
import numpy as np
import cv2
import random



logger = logging.getLogger(name=__name__)


class BaseDataTransformer:

    def __init__(self, config=None) -> None:
        self.default_config = {}
        if config is not None:
            self.config = config
        else:
            self.config = self.default_config
    
    def transform(*args, **kwargs):
        raise NotImplementedError()

class ScanNetTrainingDataTransformer(BaseDataTransformer):

    def __init__(self, config=None) -> None:
        self.default_config = {
            "group_probs" : [0.5, 0.125, 0.125, 0.125, 0.025, 0.1]
        }
        super().__init__(config=config) 

        self.transformation_group_list = [
            self.identity_transform,
            self.transfrom_group_a,
            self.transform_group_b,
            self.transform_group_c,
            self.transform_group_d,
            self.transform_group_e
        ]
        self.group_probs = self.config["group_probs"]
        assert len(self.transformation_group_list) == len(self.group_probs)
        self.probability_threshold = self.get_probability_thresholds(
            self.group_probs)

        self.rotation = RotationTransform()
        self.hflip = HorizontalFlip()
        self.vflip = VerticalFlip()
        self.brightness = AdjustBrightness()
        self.affine = AffineTransform()
    
    def identity_transform(self, input_, target):
        return input_, target

    def transform(self, input_, target):
        # Note some transfroms should only change the input (X) not Y
        # while others it should mirror in Y as well
        

        p = np.random.random()
        transformed_input, transformed_target = input_, target
        for idx, p_threshold in enumerate(self.probability_threshold):
            if p <= p_threshold:
                transformed_input, transformed_target \
                    = self.transformation_group_list[idx](input_, target)
                return transformed_input, transformed_target


    
    def get_probability_thresholds(self, probability_list):
        prob_threshold = []
        running_sum = 0
        for p in probability_list:
            running_sum += p
            prob_threshold.append(running_sum)
        
        assert prob_threshold[-1] == 1.0
        return prob_threshold

    
    def transfrom_group_a(self, input_, target):
        #probability of components of each group must sum to 1
        return self.rotation(input_, target)

    def transform_group_b(self, input_, target):
        return self.hflip(input_, target)

    def transform_group_c(self, input_, target):
        return self.vflip(input_, target)
    
    def transform_group_d(self, input_, target):
        return self.brightness(input_, target)
    
    def transform_group_e(self, input_, target):
        return self.affine(input_, target)

    def register_transformation(self, transformation):
        # TODO: to be implemented if needed
        raise NotImplementedError()

class RotationTransform:
    def __init__(self, angles=None):

        self.angles = angles
        if self.angles is None:
            self.angles = [30, 60, 90, 120, 180]

    def __call__(self, x, y):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle), None

class HorizontalFlip:
    def __init__(self, *args, **kwargs) -> None:
        #no attrs
        pass

    def __call__(self, x, y):
        return TF.hflip(x), None


class VerticalFlip:
    def __init__(self, *args, **kwargs) -> None:
        #no attrs
        pass

    def __call__(self, x, y):
        return TF.vflip(x), None

class AdjustBrightness:
    def __init__(self, brightness_factors=None) -> None:
        self.brightness_factors = brightness_factors
        if self.brightness_factors is None:
            self.brightness_factors = [0.5, 0.8, 1.2, 1.5]

    def __call__(self, x, y):
        factor = random.choice(self.brightness_factors)
        return TF.adjust_brightness(x, brightness_factor=factor), y


class AffineTransform:
    def __init__(self, angle=None, translate=None, scale=None,
                    shear=None) -> None:
        #no attrs
        self.angle = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 110, 120]
        self.translate = [[0, 1]]
        self.scale = [1.0, 1.2, 1.5, 0.8, 0.6]
        self.shear = [[15, 0], [0, 15], [15, 15], [25, 0], [0, 25], [25, 25]]

        if angle is not None:
            self.angle = angle
        if translate is not None:
            self.translate = translate
        if scale is not None:
            self.scale = scale
        if shear is not None:
            self.shear = shear


    def __call__(self, x, y):
        angle = random.choice(self.angle)
        translate = random.choice(self.translate)
        scale = random.choice(self.scale)
        shear = random.choice(self.shear)

        return TF.affine(x, angle=angle,
                translate=translate, scale=scale, shear=shear), \
                    None

#FIXME: Add changes for gravity vector once input format is finalized
