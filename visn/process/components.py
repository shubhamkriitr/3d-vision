from anyio import run_process
from visn.estimation.keypoint import OpenCvKeypointMatcher
import numpy as np
from typing import List
from visn.models import BaseGravityEstimator
from scipy.spatial.transform import Rotation
from visn.estimation.pose import PoseEstimator
from visn.benchmark.benchmark import BenchMarker
from visn.benchmark.metrics import compute_pose_error
from visn.utils import logger
from visn.process.utils import compute_alignment, compute_relative_pose
import poselib
from typing import Dict
from visn.config import read_config
import copy
# TODO : move key-names used in sample/stage data to constants 

class BasePreprocessor(object):
    """This preprocessor works on image pairs. For more than 2 images in 
    a sample, only the first two will be used.
    This class  brings all the steps of the pipeline together and provides 
    a shared execution context for the pipeline steps.
    To make the outputs of previous steps available to the next steps,
    at each of the pipeline step processes the inputs 
    (which are list of dictionaries) and
    store the results back in the same dictionary.
    """
    def __init__(self, config: Dict = {}, **kwargs) -> None:
        self._init_from_config(config)
        self.pipeline = None # to be used later (for shared context etc.)

    def _init_from_config(self, config):
        self.config = {**read_config()["preprocessor"], **config}

        # keypoint matcher
        if self.config["keypoint_matcher"] == "OpenCvKeypointMatcher":
            self.kpm = OpenCvKeypointMatcher(config={})
        else:
            raise ValueError

        # gravity estimator
        if self.config["gravity_estimator"] == "BaseGravityEstimator":
            self.gravity_estimator = BaseGravityEstimator()
        else:
            raise ValueError

        self.keypoint_threshold = self.config["keypoint_threshold"]
        self.pipeline_stage = self.config["pipeline_stage"]
            
    def process(self, batch_):
        """It process a batch of data."""
        # Assumes batch_ is a list of dictionaries 
        # {"input": [img_1, img_2, ...],
        # "K": [K_0, K_1, ....]}
        
        output = []
        
        for sample in batch_:
            self.process_one_sample(sample)
            output.append(sample)
        
        return output

    def process_one_sample(self, sample):
        """It processes one sample. `sample` is a dictionary 
        containing input images, ground truth gravity, ground truth absolute,
        pose, intrinsic matrix etc.
        {"input": [img_1, img_2, ...],
         "K": [K_0, K_1, ....],
         }
        
        It calls helper functions to do the following:
        - Keypoint detection and matching
        - Keypoint normalization
        - Gravity vector computation (if required)
        - Keypoint alignment (using gravity information)
        
        It stores all these resulting values in the same input dictionary
        `sample`.
        """
        k_inverse = [np.linalg.inv(k) for k in sample["K"]]
        input_images = sample["input_images"]
        
        if "input_keypoint_matches" not in sample:
            keypoints_0, keypoints_1 = self.kpm.get_matches(
                    input_images[0], input_images[1], self.keypoint_threshold)
        else:
            logger.info("Using keypoint matches from dataset.")
            keypoints_0, keypoints_1 = sample["input_keypoint_matches"]
            
        normalized_keypoints = self.normalized_keypoints(k_inverse,
                                    [keypoints_0, keypoints_1])
            
        if self.pipeline_stage not in sample:
            sample[self.pipeline_stage] = {}
        _stage_data = sample[self.pipeline_stage]
                
        # compute gravity if not available already:
        gravity_vectors = None
        if "input_gravity" not in sample:
            gravity_vectors = self.estimate_gravity(
                    np.concatenate([np.expand_dims(img, axis=0) 
                                    for img in input_images], axis=0)
                )
        else:
            gravity_vectors = np.array(sample["input_gravity"])
            
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
        
        if "relative_pose_gt" not in sample:
            logger.info(f"Computing relative_pose_gt")
            self.compute_relative_pose_ground_truth(sample, _stage_data)
            
        
        
        return sample # returning as well/ although output is added in place
    
    def rotate_keypoints(self,
                            rotations: List[np.ndarray],
                            images_keypoints: List[np.ndarray]):
        """Assuming keypoints.shape (N, 2) (in homogeneous coordinates) # 
        and `intrinsic_matrix` of shape (3, 3); it rotates the keypoints.
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
        It computes rotations which can move `source_vectors` to the
        corresponding `target_vectors`.
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
        """`source_vector` and `target_vector` are of shape (1, 3).
        It computes rotations which can move `source_vector` to the
        corresponding `target_vector`
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
        return compute_alignment(source_vector, target_vector)
    
    def compute_relative_pose_ground_truth(self, sample, _stage_data):
        """ 
        In the input `sample` -> `sample["input_relative_poses"]` is the
        pose of the cameras w.r.t. to some base reference (we may call
        it absolute pose). Using these
        poses, this function computes the relative pose between the two cameras
        and stores in the `_stage_data` dictionary under attribute name
        `relative_pose_gt`
        
        """
        input_relative_poses = sample["input_relative_poses"]
        
        Rt_0 = np.array(input_relative_poses[0], dtype=np.float64)
        Rt_1 = np.array(input_relative_poses[1], dtype=np.float64)
        
        Rt = compute_relative_pose(Rt_0, Rt_1)
        
        _stage_data["relative_pose_gt"] = Rt
        
        return Rt
    
    def extract_value(self, container: dict, key_sequences,
                      default_value=None):
        """ 
        `container` a dict like key value structure
        `key_sequences` a list of list of strings.
        e.g. For [["a", "b"], ["c"], ["d", "e", "f"]], 
            container["a"]["b"]
            container["c"]
            container["d"]["e"]["f"]
            will be queried in order, and the first query that resolves
            without KeyError will be returned.
            If all queries result in KeyError then `default_value`
            will be returned
        """
        value = None
        for key_seq in key_sequences:
            try:
                value = container
                for k in key_seq:
                    value = value[k]
                return value
            except KeyError:
                pass
        
        return default_value


class Preprocessor(BasePreprocessor):
    def __init__(self, config: Dict = {}, **kwargs) -> None:
        super().__init__(config, **kwargs)
    
    def process(self, batch_):
        return super().process(batch_)
    

class PoseEstimationProcessor(BasePreprocessor):
    """ 
    This class is a component of the pipeline, which computes relative pose
    solutions using the 3-point/5-point estimators by using keypoint input
    from the previous step.
    
    Look at its `process_one_sample` method for more details.
    """
    def __init__(self, config: Dict = {}, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.pipeline_stage = "_stage_pose_estimate"
    
    def _init_from_config(self, config):
        # TODO use config/ set default params
        self.estimator = PoseEstimator(config=config)
    
    def process_one_sample(self, sample):
        """ 
        `sample` is a dictionary containing input (and outputs generated so far
        in the pipeline). 
        This function does the following:
            * prepare RANSAC options, Bundle options and Camera model for
            the 3-point and 5-point estimators
            * computes the relative pose solution by calling the respective
            estimators
            * stores the resukts back in `sample["_stage_pose_estimate"]`
            dictionary to make it available to later stages of the pipeline.
        """
        if self.pipeline_stage not in sample:
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
        
        (relative_pose_Rt_3pt_up, relative_pose_Rt_3pt_up_aligned, 
        solution_metadata_3pt) = \
            self.process_3point_estimation(sample, _stage_data)
            
        relative_pose_Rt_5pt, solution_metadata_5pt = \
            self.process_5point_estimation(sample, _stage_data)
            
        _stage_data["pose"] = {
            "3pt_up": relative_pose_Rt_3pt_up,
            "3pt_up_aligned": relative_pose_Rt_3pt_up_aligned,
            "5pt": relative_pose_Rt_5pt
        }
        
        _stage_data["solution_metadata"] = {
            "3pt_up": solution_metadata_3pt,
            "5pt": solution_metadata_5pt
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
        
        solution = self.estimator.estimate_relative_pose_3pt_upright(
            x2d_0, x2d_1, cam_0, cam_1, 
            _stage_data["ransac_options"]["3pt_up"],
            _stage_data["bundle_options"]["3pt_up"]
        )
        
        
        
        pose, solution_metadata = solution
        solution_metadata = self.prune_solution_metadata(solution_metadata)
        logger.info(f"Solution metadata: "
                    f"{solution_metadata}")
        
        #pose in normalized/aligned coordinate
        Rt_norm_aligned =  pose.Rt
        
        R_align_0 = sample["_stage_preprocess"]["R_align"][0] #for first camera
        R_align_1 = sample["_stage_preprocess"]["R_align"][1] #for second
        R_na = Rt_norm_aligned[:, 0:3]
        t_na = Rt_norm_aligned[:, 3:4]
        R_norm = R_align_1.T @ R_na @ R_align_0
        t_norm = R_align_1.T @ t_na 
        # remove effect of aligning to gravity
        Rt_norm = np.concatenate([R_norm, t_norm], axis=1)
        
        
            
        return Rt_norm, Rt_norm_aligned, solution_metadata
        
        
    def process_5point_estimation(self, sample, _stage_data):
        preprocessed_data = sample["_stage_preprocess"]
        x2d_0 = preprocessed_data["normalized_keypoints"][0]
        x2d_1 = preprocessed_data["normalized_keypoints"][1]
        x2d_0 = self.to_list_of_nd_array(x2d_0)
        x2d_1 = self.to_list_of_nd_array(x2d_1)
        
        cam_0, cam_1 = _stage_data["camera_models"][0:2]
        
        solution = self.estimator.estimate_relative_pose_5pt(
            x2d_0, x2d_1, cam_0, cam_1, 
            _stage_data["ransac_options"]["5pt"],
            _stage_data["bundle_options"]["5pt"]
        )
        
        
        
        pose , solution_metadata = solution
        solution_metadata = self.prune_solution_metadata(solution_metadata)
        logger.info(f"Solution metadata: "
                    f"{solution_metadata}")
            
        return pose.Rt, solution_metadata
        
        
        
    def prepare_initial_camera_models(self, sample, _stage_data):
        """ Prepares camera models for the two images by calling the
          `self.estimator`
        """
        cam_0 = self.estimator.prepare_cam()
        cam_1 = self.estimator.prepare_cam()
        return cam_0, cam_1
    
    def prepare_ransac_options(self, sample, _stage_data):
        
        ransac_option_3pt_up = self.estimator.prepare_ransac_options()
        ransac_option_5pt = self.estimator.prepare_ransac_options()
        return ransac_option_3pt_up, ransac_option_5pt
    
    def prepare_bundle_options(self, sample, _stage_data):
        
        bundle_option_3pt_up = self.estimator.prepare_bundle_options()
        bundle_option_5pt = self.estimator.prepare_bundle_options()
        return bundle_option_3pt_up, bundle_option_5pt
        

    def to_list_of_nd_array(self, x):
        x_ = [] # TODO: replace with an optimal version
        for i in range(x.shape[0]):
            x_.append(x[i])
        return x_
    
    def prune_solution_metadata(self, solution_metadata):
        """ 
        `solution_metadata` contains RANSAC run information, like inliers,
        number of inliers, number of iterations to converge etc.
        This method keeps only relevant information and removes `inliers` 
        attribute as it is not required for report generation.
        """
        return {k:v for k, v in solution_metadata.items() if k != "inliers"}



class BenchmarkingProcessor(BasePreprocessor):
    """This class brings together metrics functions and runtime measurement
    function, and uses the pose estimations and ground truth data to 
    record pose errors. Using runtime measurement function it also
    calls the pose estimators again for multiple number of times (>>>currently
    set to 20) to get the average run time, and logs it.
    """
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
        self.benchmarker = BenchMarker()
        self.pipeline_stage = "_stage_benchmark"
    
    def process_one_sample(self, sample):
        """ It use previously computed data like aligned keypoints etc.
        which is available in the input `sample` (dictionary) to call
        the 3-point/5-point relative pose algorithms again to estimate their
        average runtime for the given sample. Besides this it also calls
        `log_pose_errors` to store pose error and runtime information in the 
        `sample["_stage_benchmark]` dictionary.  
        
        Also see: `log_pose_errors`
        """
        # runtimes in nano seconds
        args_kwargs_for_3pt_up = [
            # args
            (
            sample["_stage_preprocess"]["normalized_aligned_keypoints"][0],
            sample["_stage_preprocess"]["normalized_aligned_keypoints"][1],
            sample["_stage_pose_estimate"]["camera_models"][0],
            sample["_stage_pose_estimate"]["camera_models"][1],
            sample["_stage_pose_estimate"]["ransac_options"]["3pt_up"],
            sample["_stage_pose_estimate"]["bundle_options"]["3pt_up"]
            ),
            #kwargs
            {}
        ]
        
        args_kwargs_for_5pt = [
            # args
            (
            sample["_stage_preprocess"]["normalized_keypoints"][0],
            sample["_stage_preprocess"]["normalized_keypoints"][1],
            sample["_stage_pose_estimate"]["camera_models"][0],
            sample["_stage_pose_estimate"]["camera_models"][1],
            sample["_stage_pose_estimate"]["ransac_options"]["5pt"],
            sample["_stage_pose_estimate"]["bundle_options"]["5pt"]
            ),
            #kwargs
            {}
        ]
        
        
        
        runtimes_ns = self.benchmarker.compare_runtimes(
            methods=[poselib.estimate_relative_pose_3pt_upright,
                     poselib.estimate_relative_pose],
            args_kwargs=[args_kwargs_for_3pt_up, args_kwargs_for_5pt],
            use_same_args_kwargs_for_all=False,
            num_trials=20,
            normalization_mode=None
        )
        
        ratio = self.benchmarker.normalize_values(runtimes_ns,
                                                  mode="first_element")
        
        logger.info(f"Runtimes for 3pt_upright, 5pt = {runtimes_ns} , "
                    f"ratio = {ratio}")
        
        sample[self.pipeline_stage] = {
            "runtimes_ns": runtimes_ns,
            "runtime_ratio": ratio
        }
        
        self.log_pose_errors(sample, sample[self.pipeline_stage])
        if ("_stage_angle_manipulator" not in sample):
            logger.info(f"{self.pipeline_stage} : {sample[self.pipeline_stage]}")
        else :
            angl_err = sample["_stage_angle_manipulator"]
            logger.info(f"{self.pipeline_stage} : {sample[self.pipeline_stage]} {angl_err}")
        return sample
    
    def log_pose_errors(self, sample, _stage_data: dict):
        """ 
        Computes pose error between groundtruth pose and the estimated poses.
        Stores he estimated error in the `_stage_data` provided.
        """
        estimated = sample["_stage_pose_estimate"]
        preprocessed = sample["_stage_preprocess"]
        
        #check ground truth pose in input
        Rt_gt = self.extract_value(estimated,
                [["input_relative_pose_gt"],
                 ["_stage_input", "relative_pose_gt"],
                ], default_value=None)
        if Rt_gt is None:
            logger.warning(f"Could not find `relative_pose_gt` in inputs,"
                         f"Will check preprocessing stage outputs now.")
        
            Rt_gt = self.extract_value(preprocessed, [["relative_pose_gt"]],
                                       default_value=None)
            
        if Rt_gt is None:
            logger.error(f"Ground truth relative pose is not available. "
                         f"Skipping pose error computation.")
            return
        
        Rt_3pt_up = estimated["pose"]["3pt_up"]
        Rt_5pt = estimated["pose"]["5pt"]
        
        rotation_err_3pt_up, t_error_3pt_up = \
            compute_pose_error(Rt_3pt_up, Rt_gt, degrees=True)
        
        rotation_err_5pt, t_error_5pt = \
            compute_pose_error(Rt_5pt, Rt_gt, degrees=True)
        
        _stage_data["pose_error_rotation"] = {
            "3pt_up": rotation_err_3pt_up,
            "5pt": rotation_err_5pt
        }
        
        _stage_data["pose_error_translation"] = {
            "3pt_up": t_error_3pt_up,
            "5pt": t_error_5pt
        }
    
    @property 
    def _schema(self):
        """It justs visually documents the output structure, that is added
        to the input `sample` (check `process_one_sample` method for details.)
        """
        return {
            self.pipeline_stage  : {
                "runtimes_ns": None,
                "runtime_ratio": None,
                "pose_error_rotation" : {
                    "3pt_up": None,
                    "5pt": None
                },
                "pose_error_translation" : {
                    "3pt_up": None,
                    "5pt": None
                }
                
            }
        }
        
        
class AngleManipulatorProcessor(BasePreprocessor):
    """It creates (perturbed) gravity estimates with varying degrees of errors
    using interpolation.
    
    """
    
    def __init__(self, config: Dict = ..., **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.target_angles = config["target_angles"]
        self.run_process = config["run_process"]
        self.pipeline_stage = "_stage_angle_manipulator"
    
    def _init_from_config(self, config):
        pass

    def process(self, batch_):
            # Assumes batch_ is a list of dictionaries 
            # {"input": [img_1, img_2, ...],
            # "K": [K_0, K_1, ....]}
            
            output = []
            
            for sample in batch_:
                new_samples = self.process_one_sample(sample)
                for s in new_samples:
                    output.append(s)
            
            return output

    def process_one_sample(self, sample):
        
        new_samples = []
        if (sample["use_gravity_pred"] == False) or not(self.run_process):
            new_samples.append(sample)
            return new_samples

        for angl in (self.target_angles):
            new_s = copy.deepcopy(sample)
            if self.pipeline_stage not in new_s:
                new_s[self.pipeline_stage] = {}
            _stage_data = new_s[self.pipeline_stage]
            _stage_data["input_gravity_err"] = angl

            # if angle is minus dont change the gravity input
            if angl<0 :
                new_samples.append(new_s)
                continue

            # if angle not minus
            for i in range(len(new_s['input_gravity'])):
                new_s["input_gravity"][i] = self._interpolate_angle(
                    new_s['input_gravity_gt'][i],
                    new_s['input_gravity_pred'][i],
                    np.pi/180*angl
                )
            new_samples.append(new_s)
            
        return new_samples
    
    def _rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        if s==0:
            return  np.eye(3)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    def _rotation_z_allign(self, vec):
        return self._rotation_matrix_from_vectors(vec,np.array([0, 0, 1]))

    def _interpolate_angle(self, ref, trg, angle):
        m_a = np.linalg.norm(ref)
        m_b = np.linalg.norm(trg)
        a = (ref / m_a).reshape(3)
        b = (trg / m_b).reshape(3)
        v = np.cross(a, b)
        mat = self._rotation_z_allign(v)
        r1 = np.array([[np.cos(angle), -np.sin(angle), 0], 
                    [np.sin(angle),  np.cos(angle), 0], 
                    [           0,               0, 1]])
        r2 = np.array([[np.cos(-angle), -np.sin(-angle), 0], 
                    [np.sin(-angle),  np.cos(-angle), 0], 
                    [             0,               0, 1]])
        s1 = ( mat.T @ r1 @ mat @ a.reshape(3,1) ).reshape(3)
        s2 = ( mat.T @ r2 @ mat @ a.reshape(3,1) ).reshape(3)
        check = np.dot(s1, b) > np.dot(s2, b)
        return s1*m_b if check else s2*m_b
        
    
if __name__ == "__main__":
    proc = Preprocessor()
    
    s = np.array([[1, 1, 1]])
    t = np.array([[1, 1, 1]])
    
    v, theta = proc.compute_alignment(s, t)
    
    print(v, theta)
    

