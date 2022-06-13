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

DEFAULT_POSELIB_BUNDLE_OPTIONS = {
    'max_iterations': 2, 'loss_scale': 1.0, 'loss_type': 'CAUCHY',
    'gradient_tol': 1e-10, 'step_tol': 1e-08, 'initial_lambda': 0.001,
    'min_lambda': 1e-10, 'max_lambda': 10000000000.0, 'verbose': False}


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
        # TODO: Adjust based on data
        return {'model': model, 'width': width,
                'height': height, 'params': params}
    
    def prepare_ransac_options(self, **kwargs):
        ransac_options = copy.deepcopy(DEFAULT_POSELIB_RANSAC_OPTIONS)
        for k in kwargs:
            if k in ransac_options:
                ransac_options[k] = kwargs[k]
            else:
                logger.warning(f"Keyword argument {k} is unused/unrecognized.")
        return ransac_options
    
    def prepare_bundle_options(self, **kwargs):
        bundle_options = copy.deepcopy(DEFAULT_POSELIB_BUNDLE_OPTIONS)
        for k in kwargs:
            if k in bundle_options:
                bundle_options[k] = kwargs[k]
            else:
                logger.warning(f"Keyword argument {k} is unused/unrecognized.")
        return bundle_options
    
    
    @property
    def estimate_relative_pose_5pt(self):
        # Use this for benchmarking
        return poselib.estimate_relative_pose
    
    @property
    def estimate_relative_pose_3pt_upright(self):
        return poselib.estimate_relative_pose_3pt_upright

        
if __name__ == "__main__":
    pass
    
        