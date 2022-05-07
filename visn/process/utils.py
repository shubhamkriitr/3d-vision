import numpy as np
from scipy.spatial.transform import Rotation

def compute_alignment(source_vector, target_vector):
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

def compute_alignment_rotation(source_vector, target_vector):
    """`source_vector` and `target_vector` are of shape (1, 3)
    """
    normal_unit_vector, theta = compute_alignment(
        source_vector, target_vector)
    
    # theta is the magnitude
    R = Rotation.from_rotvec(theta*normal_unit_vector)
    R = R.as_matrix() # it would be of shape (1, 3, 3)
    
    return R[0] # shape (3, 3)

def compute_relative_pose(reference_absolute_pose, target_absolute_pose):
    
    R1, t1 = reference_absolute_pose[:, 0:3], reference_absolute_pose[:, 3:4]
    R2, t2 = target_absolute_pose[:, 0:3], target_absolute_pose[:, 3:4]

    R = R2 @ R1.T
    t = t2 - R @ t1    
    
    Rt = np.concatenate([R, t], axis=1)
    
    return Rt