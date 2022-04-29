from visn.solvers.keypoint import OpenCvKeypointMatcher
import numpy as np
from typing import List
from visn.models import BaseGravityEstimator
from scipy.spatial.transform import Rotation

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
    
    def compute_alignment_rotation(self, source_vector, target_vector):
        normal_unit_vector, theta = self.compute_alignment(source_vector,
                                                           target_vector)
        
        # angle of rotation (`theta`) is required to be the magnitude
        R = Rotation.from_rotvec(theta*normal_unit_vector) 
        
        return R.as_matrix() # check shape before using
        # it can be either (3, 3) or (n, 3, 3) based on the input shape
    
    def compute_alignment(self, source_vector, target_vector):
        normal_vector = np.cross(source_vector, target_vector)
        normal_unit_vector = normal_vector/np.linalg.norm(normal_vector)
        cos_theta = np.dot(normal_vector, 
                           target_vector.T)/(np.linalg.norm(target_vector))
        cos_theta = np.clip(cos_theta, a_min=0., a_max=1.)
        theta = np.arccos(cos_theta)
        
        return normal_unit_vector, theta
        
        
