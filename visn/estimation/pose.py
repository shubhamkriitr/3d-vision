import poselib
import os
import numpy as np
from visn.data.synthetic import (CameraPairDataGenerator,
    CameraPairDataGeneratorUpright3Pt)
from visn.estimation.keypoint import OpenCvKeypointMatcher
from visn.utils import logger
import matplotlib.pyplot as plt
from visn.benchmark.timing import benchmark_runtime
import copy

default_ransac_options = poselib.RansacOptions()
default_bundle_options = poselib.BundleOptions()

# Default bundle adjustmnet setting
DEFAULT_POSELIB_BUNDLE_OPTIONS = {
    'max_iterations': 2, 'loss_scale': 1.0, 'loss_type': 'CAUCHY',
    'gradient_tol': 1e-10, 'step_tol': 1e-08, 'initial_lambda': 0.001,
    'min_lambda': 1e-10, 'max_lambda': 10000000000.0, 'verbose': False}

# Default RANSAC setting
DEFAULT_POSELIB_RANSAC_OPTIONS = {
    'max_iterations': 100000, 'min_iterations': 2,
    'dyn_num_trials_mult': 3.0, 'success_prob': 0.95,
    'max_reproj_error': 12.0, 'max_epipolar_error': 1e-2,
    'seed': 0, 'progressive_sampling': False,
    'max_prosac_iterations': 100000}
    
class PoseEstimator(object):
    
    def __init__(self, config) -> None:
        self.config = config
    
    def prepare_cam(self, model='SIMPLE_PINHOLE', 
                width=1200, height=800,  params=[1, 0, 0]):
        """
        Generate camera model for poseLib
        INput:
            `width`
            `height`
            `param`: list of number (4 or 3 numbers)
                3 numbers [f_xy, center_x, center_y]
                4 numbers [f_x, f_y, center_x, center_y]
                e.g. normalized coordinate [1,0,0]
        """
        # TODO: Adjust based on data
        return {'model': model, 'width': width,
                'height': height, 'params': params}
    
    def prepare_ransac_options(self, **kwargs):
        """
        Generate setting for the psoelib RANSAC.
        input:
            `kwargs`: the setting for ransac. the parameter list can be 
                seen in DEFAULT_POSELIB_RANSAC_OPTIONS above
        """
        ransac_options = copy.deepcopy(DEFAULT_POSELIB_RANSAC_OPTIONS)
        for k in kwargs:
            if k in ransac_options:
                ransac_options[k] = kwargs[k]
            else:
                logger.warning(f"Keyword argument {k} is unused/unrecognized.")
        return ransac_options
    
    def prepare_bundle_options(self, **kwargs):
        """
        Generate setting for the poselib bundle adjustmnet.
        input:
            `kwargs`: the setting for bundle adjustment poselib. the parameter list can be 
                seen in DEFAULT_POSELIB_BUNDLE_OPTIONS above
        """
        bundle_options = copy.deepcopy(DEFAULT_POSELIB_BUNDLE_OPTIONS)
        for k in kwargs:
            if k in bundle_options:
                bundle_options[k] = kwargs[k]
            else:
                logger.warning(f"Keyword argument {k} is unused/unrecognized.")
        return bundle_options
    
    
    @property
    def estimate_relative_pose_5pt(self):
        """
        This is the wrapper to 5-point Poselib RANSAC (bundle asjustment aslo includes)
        """
        # TODO: make sure that this wrapping does not affect the comparison
        return poselib.estimate_relative_pose
    
    @property
    def estimate_relative_pose_3pt_upright(self):
        """
        This is the wrapper to 3s-point Poselib RANSAC (bundle asjustment aslo includes)
        """
        return poselib.estimate_relative_pose_3pt_upright


    
        