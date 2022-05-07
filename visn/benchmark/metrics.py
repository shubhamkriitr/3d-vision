import numpy as np
import math
from visn.utils import logger

def compute_rotation_pose_error(R1_estimated, R_true):
    R_prod = np.dot(R1_estimated, np.transpose(R_true))
    
    cos_theta = max( min( 1.0, 0.5 * (np.trace(R_prod) - 1.0) ),  -1.0)
    
    rotation_err = math.acos(cos_theta)
    
    if np.sum(np.isnan(rotation_err)):
        logger.warning(f"Unexpected Nan values \n "
                     f"R1_estimated={R1_estimated} \n R_true={R_true}")
    
    return rotation_err


def compute_translation_pose_error(t_est, t_true):
    t_est = t_est.flatten()
    t_true = t_true.flatten()
    eps = 1e-15
    
    t_est = t_est / (np.linalg.norm(t_est) + eps)
    t_true = t_true / (np.linalg.norm(t_true) + eps)
    
    loss = np.maximum(eps, (1.0 - np.sum(t_est * t_true)**2))
    error = np.arccos(np.sqrt(1 - loss))
    
    if np.sum(np.isnan(error)):
        logger.warning(f"Unexpected nan values.\n"
                       f"t_estimated={t_est} \n t_true = {t_true}")
        
    return error
    

def compute_pose_error(Rt_estimated, Rt_ground_truth, degrees=False):
    """
    `Rt_estimated` and `Rt_ground_truth` are 3 x 4 projection matrices.
    """
    R_e, R_gt = Rt_estimated[:, 0:3], Rt_ground_truth[:, 0:3]
    t_e, t_gt = Rt_estimated[:, 3:4], Rt_ground_truth[:, 3:4]
    
    rotation_err = compute_rotation_pose_error(R_e, R_gt)
    translation_err = compute_translation_pose_error(t_e, t_gt)
    
    if degrees:
        rotation_err = np.degrees(rotation_err)
        translation_err = np.degrees(translation_err)
    
    return rotation_err, translation_err