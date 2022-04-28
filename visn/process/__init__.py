from visn.solvers.keypoint import OpenCvKeypointMatcher
import numpy as np
from typing import List
from models import BaseGravityEstimator

class BasePreprocessor(object):
    """This preprocessor works on image pairs. For more than 2 images in 
    a sample, only the first two will be used.
    """
    def __init__(self, config = None, **kwargs) -> None:
        self._init_from_config(config)
        self.pipeline = None # to be used later (for shared context etc.)
        
    
    def _init_from_config(self, config):
        if config is None:
            self.config = {
                "keypoint_matcher": "OpenCvKeypointMatcher", # use factory: TODO,
                "gravity_estimator": "BaseGravityEstimator"
            }
        else:
            self.config = config
            
        assert self.config["keypoint_matcher"] == "OpenCvKeypointMatcher"
        self.kpm = OpenCvKeypointMatcher(config={})
        self.gravity_estimator = BaseGravityEstimator()
        self.keypoint_threshold = 0.25
            
    def process(self, batch_):
        # Assumes batch_ is a list of dictionaries 
        # {"input": [img_1, img_2, ...],
        # "K": [K_0, K_1, ....]}
        
        output = []
        
        for sample in batch_:
            k_inverse = [np.linalg.inv(k) for k in sample["K"]]
            input_images = sample["input_images"]
            
            keypoints_0, keypoints_1 = self.kpm.get_matches(
                input_images[0], input_images[1], self.keypoint_threshold)
            
            normalized_keypoints = self.normalized_keypoints(k_inverse,
                                    [keypoints_0, keypoints_1])
            
            gravity_vectors = self.estimate_gravity(
                np.concatenate([np.expand_dims(img, axis=0) 
                                for img in input_images], axis=0)
            )
            
            
            sample["_stage_preprocess"] = {
                "K_inverse": k_inverse,
                "keypoints": [keypoints_0, keypoints_1],
                "normalized_keypoints": normalized_keypoints ,
                "gravity_vectors": gravity_vectors
            }
            
            output.append(sample)
        
        return output

    def normalized_keypoints(self,
                            k_inv_list: List[np.ndarray],
                            images_keypoints: List[np.ndarray]):
        """Assuming keypoints.shape (N, 2) # 
        and `intrinsic_matrix` os shape (3, 3)
        """
        output = []
        for image_idx in range(len(images_keypoints)):
            keypoints = images_keypoints[image_idx]
            intrinsic_matrix_inverse = k_inv_list[image_idx]
            kp_homogeneous = np.concatenate(
                [keypoints,np.ones(shape=(keypoints.shape[0], 1)
                                    )], axis=1)
            result = kp_homogeneous@intrinsic_matrix_inverse.T
            
            result = result / result[:, 2:3] # rescale
            output.append(result[:, 0:2])  # non-homogenous
        
        return output
    
    @property
    def estimate_gravity(self):
        return self.gravity_estimator.estimate_gravity
        
        
