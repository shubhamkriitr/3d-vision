from visn.estimation.keypoint import OpenCvKeypointMatcher
import numpy as np
from typing import List
from visn.models import BaseGravityEstimator
from scipy.spatial.transform import Rotation
from visn.estimation.pose import PoseEstimator
from visn.utils import logger
# TODO : move key-names used in sample/stage data to constants 

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
        self.pipeline_stage = "_stage_preprocess"
            
    def process(self, batch_):
        # Assumes batch_ is a list of dictionaries 
        # {"input": [img_1, img_2, ...],
        # "K": [K_0, K_1, ....]}
        
        output = []
        
        for sample in batch_:
            self.process_one_sample(sample)
            output.append(sample)
        
        return output

    def process_one_sample(self, sample):
        k_inverse = [np.linalg.inv(k) for k in sample["K"]]
        input_images = sample["input_images"]
            
        keypoints_0, keypoints_1 = self.kpm.get_matches(
                input_images[0], input_images[1], self.keypoint_threshold)
            
        normalized_keypoints = self.normalized_keypoints(k_inverse,
                                    [keypoints_0, keypoints_1])
            
        if self.pipeline_stage not in sample:
            sample[self.pipeline_stage] = {}
        _stage_data = sample[self.pipeline_stage]
                
        # compute gravity if not available already:
        gravity_vectors = None
        if "gravity_vectors" not in _stage_data:
            gravity_vectors = self.estimate_gravity(
                    np.concatenate([np.expand_dims(img, axis=0) 
                                    for img in input_images], axis=0)
                )
            
            
        _stage_data["K_inverse"] = k_inverse
        _stage_data["keypoints"] = [keypoints_0, keypoints_1]
        _stage_data["normalized_keypoints"] = normalized_keypoints
        _stage_data["gravity_vectors"] = gravity_vectors
        
        
        # Align vectors # TODO: check pipeline context to see if it needs
        # to  be computed
        pipeline_requires_alignment = True

        if pipeline_requires_alignment:
            target_vectors = np.zeros_like(gravity_vectors)
            target_vectors[:, 1] = 1. # y-axis
            alignment_rotations = self.compute_alignmet_rotations(
                source_vectors=gravity_vectors,
                target_vectors=target_vectors
            )
            _stage_data["R_align"] = alignment_rotations
            
            # list of normalized and aligned keyppoints
            # arrays for each of the images in the group (or pair)
            _stage_data["normalized_aligned_keypoints"] = \
                self.rotate_keypoints(alignment_rotations,
                                      normalized_keypoints)
            
        
        
        return sample # returning as well/ although output is added in place
    
    def rotate_keypoints(self,
                            rotations: List[np.ndarray],
                            images_keypoints: List[np.ndarray]):
        """Assuming keypoints.shape (N, 2) (in homogeneous coordinates) # 
        and `intrinsic_matrix` os shape (3, 3)
        """
        output = []
        for image_idx in range(len(images_keypoints)):
            keypoints = images_keypoints[image_idx]
            r = rotations[image_idx]
            
            # to homogeneous
            keypoints = np.concatenate(
                [keypoints,np.ones(shape=(keypoints.shape[0], 1)
                                    )], axis=1)
            result = keypoints@r.T # result of shape (N, 3)
            
            result = result / result[:, 2:3] # rescale
            output.append(result[:, 0:2])  # non-homogenous
        
        return output

    def normalized_keypoints(self,
                            k_inv_list: List[np.ndarray],
                            images_keypoints: List[np.ndarray]):
        """Assuming images_keypoints.shape (N, 2) # 
        and `k_inv_list` to be a list of K^-1 with shape (3, 3)
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
    
    def compute_alignmet_rotations(self, source_vectors, target_vectors):
        """
        `source_vectors` and `target_vectors` are of shape (B, 3). (where 
        `B` is  the batch size)
        """
        B = source_vectors.shape[0]
        rotations = []
        for idx in range(B):
            r = self._compute_alignment_rotation(
                source_vector=source_vectors[idx:idx+1],
                target_vector=target_vectors[idx:idx+1]
            )
            rotations.append(r)
        
        return rotations
    
    def _compute_alignment_rotation(self, source_vector, target_vector):
        """`source_vector` and `target_vector` are of shape (1, 3)
        """
        normal_unit_vector, theta = self.compute_alignment(
            source_vector, target_vector)
        
        # theta is the magnitude
        R = Rotation.from_rotvec(theta*normal_unit_vector)
        R = R.as_matrix() # it would be of shape (1, 3, 3)
        
        return R[0] # shape (3, 3)
    
    def compute_alignment(self, source_vector, target_vector):
        """`source_vector` and `target_vector` are of shape (1, 3)
        """
        normal_vector = np.cross(source_vector, target_vector)
        if np.allclose(normal_vector, 0., rtol=1e-8, atol=1e-8):
            normal_vector = np.zeros_like(source_vector)
            nonzero_at, = np.where(np.logical_not(np.isclose(
                                source_vector[0], 0., rtol=1e-8, atol=1e-8)))

            if len(nonzero_at) > 1: 
                idx0, idx1 = nonzero_at[0], nonzero_at[1]
                normal_vector[0][idx0] = source_vector[0][idx1]
                normal_vector[0][idx1] = -source_vector[0][idx0]
            elif len(nonzero_at) == 1: # x, y or z axis
                normal_vector[0][(nonzero_at[0]+1)%3] = 1.0
            else: # zero vectors
                normal_vector[0][1] = 1.0 # choose y-axis
            
            
        normal_unit_vector = normal_vector/np.linalg.norm(normal_vector)
        cos_theta = np.dot(source_vector, target_vector.T)/(
            np.linalg.norm(source_vector)*np.linalg.norm(target_vector))
        cos_theta = np.clip(cos_theta, a_min=-1., a_max=1.)
        theta = np.arccos(cos_theta)
        
        return normal_unit_vector, theta
        
class Preprocessor(BasePreprocessor):
    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config, **kwargs)
    
    def process(self, batch_):
        return super().process(batch_)
    

class PoseEstimationProcessor(BasePreprocessor):
    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.pipeline_stage = "_stage_pose_estimate"
    
    def _init_from_config(self, config):
        # TODO use config/ set default params
        self.estimator = PoseEstimator(config=config)
    
    def process_one_sample(self, sample):
        
        sample[self.pipeline_stage] = {}
        _stage_data = sample[self.pipeline_stage]
        
        ransac_3pt_up, ransac_5pt = self.prepare_ransac_options(sample, _stage_data)
        bundle_3pt_up, bundle_5pt = self.prepare_bundle_options(sample, _stage_data)
        cam_0, cam_1 = self.prepare_initial_camera_models(sample, _stage_data)
        
        _stage_data["ransac_options"] = {"3pt_up": ransac_3pt_up,
                                         "5pt": ransac_5pt}
        _stage_data["bundle_options"] = {
            "3pt_up": bundle_3pt_up,
            "5pt": bundle_5pt
        }
        _stage_data["camera_models"] = [cam_0, cam_1]
        
        relative_pose_Rt_3pt_up = \
            self.process_3point_estimation(sample, _stage_data)
            
        relative_pose_Rt_5pt = \
            self.process_5point_estimation(sample, _stage_data)
            
        _stage_data["pose"] = {
            "3pt_up": relative_pose_Rt_3pt_up,
            "5pt": relative_pose_Rt_5pt
        }
        
        
        return sample
    
    def process_3point_estimation(self, sample, _stage_data):
        """ 
        
        `_stage_data` is the part of the running dict (`sample`) which
        the current stage is modifying. `sample` is the full running
        dict (being passed along the steps of the pipeline)
        """
        preprocessed_data = sample["_stage_preprocess"]
        x2d_0 = preprocessed_data["normalized_aligned_keypoints"][0]
        x2d_1 = preprocessed_data["normalized_aligned_keypoints"][1]
        
        x2d_0 = self.to_list_of_nd_array(x2d_0)
        x2d_1 = self.to_list_of_nd_array(x2d_1)
        cam_0, cam_1 = _stage_data["camera_models"][0:2]
        
        possible_poses = self.estimator.estimate_relative_pose_3pt_upright(
            x2d_0, x2d_1, cam_0, cam_1, 
            _stage_data["ransac_options"]["3pt_up"],
            _stage_data["bundle_options"]["3pt_up"]
        )
        
        if len(possible_poses) > 1:
            logger.warning(f"{len(possible_poses)} possible solutions found"
                           f". Choosing the first")
        
        pose = possible_poses[0]
            
        return pose.Rt
        
        
    def process_5point_estimation(self, sample, _stage_data):
        preprocessed_data = sample["_stage_preprocess"]
        x2d_0 = preprocessed_data["normalized_keypoints"][0]
        x2d_1 = preprocessed_data["normalized_keypoints"][1]
        x2d_0 = self.to_list_of_nd_array(x2d_0)
        x2d_1 = self.to_list_of_nd_array(x2d_1)
        
        cam_0, cam_1 = _stage_data["camera_models"][0:2]
        
        possible_poses = self.estimator.estimate_relative_pose_5pt(
            x2d_0, x2d_1, cam_0, cam_1, 
            _stage_data["ransac_options"]["5pt"],
            _stage_data["bundle_options"]["5pt"]
        )
        
        if len(possible_poses) > 1:
            logger.warning(f"{len(possible_poses)} possible solutions found"
                           f". Choosing the first")
        
        pose = possible_poses[0]
            
        return pose.Rt
        
        
        
    def prepare_initial_camera_models(self, sample, _stage_data):
        # TODO: may use data in sample/_stage_data .. and add
        # camera model dicts
        cam_0 = self.estimator.prepare_cam()
        cam_1 = self.estimator.prepare_cam()
        return cam_0, cam_1
    
    def prepare_ransac_options(self, sample, _stage_data):
        # TODO: may use data in sample/_stage_data .. and add
        # ransac parameters information
        ransac_option_3pt_up = self.estimator.prepare_ransac_options()
        ransac_option_5pt = self.estimator.prepare_ransac_options()
        return ransac_option_3pt_up, ransac_option_5pt
    
    def prepare_bundle_options(self, sample, _stage_data):
        # TODO: may use data in sample/_stage_data .. and add
        # bundle parameters information
        bundle_option_3pt_up = self.estimator.prepare_bundle_options()
        bundle_option_5pt = self.estimator.prepare_bundle_options()
        return bundle_option_3pt_up, bundle_option_5pt
        

    def to_list_of_nd_array(self, x):
        x_ = [] # TODO: replace with an optimal version
        for i in range(x.shape[0]):
            x_.append(x[i])
        return x_



class BenchmarkingProcessor(BasePreprocessor):
    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config, **kwargs)
        if "pipeline" in kwargs:
            self.pipeline = kwargs["pipeline"]
        
    def _init_from_config(self, config):
        if config is None:
            self.config = {
            }
        else:
            self.config = config
    
    def process_one_sample(self, sample):
        return sample
    
if __name__ == "__main__":
    proc = Preprocessor()
    
    s = np.array([[1, 1, 1]])
    t = np.array([[1, 1, 1]])
    
    v, theta = proc.compute_alignment(s, t)
    
    print(v, theta)
    

