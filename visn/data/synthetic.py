import numpy as np
from scipy.spatial.transform import Rotation
import os
import shutil
import matplotlib

class SyntheticDataGenerator(object):
    def __init__(self) -> None:
        pass
    

class CameraPairDataGenerator(SyntheticDataGenerator):
    def __init__(self, config={}) -> None:
        super(CameraPairDataGenerator, self).__init__()
        self.config = config # not being used currently TODO
    
    def get_all(self, num_samples=None, rotation=None):
        if rotation is None:
            rotation = self.get_rotation_matrix()
        camera_center_translation = np.array([[10], [20], [10]], dtype=np.float64)
        # >>> translation = - rotation @ camera_center_translation
        # is for the case when translation is done before rotation
        # but looking at the results Poselib assumes Rotation then translation
        translation = camera_center_translation
        t_cross = self.get_a_cross(translation)
        essential_matrix = t_cross @ rotation
        
        x1, x2 = self.generate_correspondences(rotation, translation,
                                               num_samples)
        return x1, x2, rotation, translation, essential_matrix
    
    def get_rotation_matrix(self):
        rotation = Rotation.from_euler('zyx', [90, 0, 0], degrees=True)
        rotation = rotation.as_matrix().astype(np.float64)
        return rotation
    
    def generate_correspondences(self, rotation_matrix, translation,
                                 num_samples=None):
        r = rotation_matrix
        t = translation
        p = np.concatenate([r, t], axis=1)
        
        # >>> x1 = np.random.randint(low=1, high=10, size=(num_samples, 3))
        x1 = np.array(
            [[0, 0, 1],
             [0, 1, 2],
             [0, 2, 3],
             [1, 2, 4],
             [2, 2, 5]], dtype=np.float64
        ) # num_samples x 3
        
        # TODO: add random sampling
        if num_samples is None:
           x1 = np.array(
            [[0, 0, 1],
             [0, 1, 2],
             [0, 2, 3],
             [1, 2, 4],
             [2, 2, 5],
             [5, 5, 6],
             [4, 4, 7],
             [4, 8, 8]], dtype=np.float64
             ) 
           num_samples = x1.shape[0]
        
        ones = np.ones(shape=(x1.shape[0], 1), dtype=x1.dtype)
        
        x3d = np.concatenate([x1, ones], axis=1)
        x2 = (p @ x3d.T ).T
        
        x2 = x2[:, :]/x2[:, 2:3]
        
        if x1.shape[0] > num_samples:
            x1 = x1[0:num_samples]
            x2 = x2[0:num_samples]
        
        return x1, x2
    
    def get_a_cross(self, a):
        a = np.ravel(a)
        if a.shape[0] != 3:
            raise AssertionError(f"`a` must be a 3 vector")
        return np.array([[0, -a[2], a[1]],
                         [a[2], 0, -a[0]],
                         [-a[1], a[0], 0]], dtype=np.float64)
        
def FOVBoudaryPlanes(fov_x, fov_y, degress=True):
    # 1 x 3
    b0 = np.array([[ 1, 0,0]])
    b1 = np.array([[-1, 0,0]])
    b2 = np.array([[ 0, 1,0]])
    b3 = np.array([[ 0,-1,0]])
    # 3 x 3
    r0 = Rotation.from_euler('xyz', [0, -fov_y/2, 0], degrees=degress)
    r1 = Rotation.from_euler('xyz', [0,  fov_y/2, 0], degrees=degress)
    r2 = Rotation.from_euler('xyz', [-fov_x/2, 0, 0], degrees=degress)
    r3 = Rotation.from_euler('xyz', [ fov_x/2, 0, 0], degrees=degress)

    # 3 x 1
    b0 = r0@b0.T
    b1 = r1@b1.T
    b2 = r2@b2.T
    b3 = r3@b3.T
    # 3 x 4
    return np.concatenate([b0,b1,b2,b3], axis=1)

def TransformPoints(x1, r, t):
        p = np.concatenate([r, t], axis=1)
        ones = np.ones(shape=(x1.shape[0], 1), dtype=x1.dtype)
        x3d = np.concatenate([x1, ones], axis=1)
        x2 = (p @ x3d.T ).T
        
        x2 = x2[:, :]/x2[:, 2:3]
        
        return x2[:,:3]

def CheckIsInFOV(points, fov_x, fov_y):
    bound = FOVBoudaryPlanes(fov_x, fov_y)
    return np.sum(points*bound > 0, axis = 1)




class CameraPairDataGeneratorUpright3Pt(CameraPairDataGenerator):
    def __init__(self, config={}) -> None:
        super().__init__(config)
    
    def get_rotation_matrix(self):
        # Poselib assumes the rotation is about y axis for upright 3 point case
        rotation = Rotation.from_euler('zyx', [0, 90, 0], degrees=True)
        rotation = rotation.as_matrix().astype(np.float64)
        return rotation


DIR_CALIBRATION = "calibration"
DIR_GRAVITY_GT = "gravity_gt"
DIR_GRAVITY_PRED = "gravity_pred"
DIR_IMAGES = "images"
DIR_REL_POSE = "relative_pose"
DIR_ROLL_PITCH_GT = "roll_pitch_gt"
DIR_ROLL_PITCH_PRED = "roll_pitch_pred"

ALL_VISN_DIRS = [
    DIR_CALIBRATION,
    DIR_GRAVITY_GT,
    DIR_GRAVITY_PRED,
    DIR_IMAGES,
    DIR_REL_POSE,
    DIR_ROLL_PITCH_GT,
    DIR_ROLL_PITCH_PRED
]
class SyntheticVisnDataGenerator(object):
    
    def __init__(self) -> None:
        pass
    
    def generate(num_pairs, output_dir, *args, **kwargs):
        pass
    
    def save_single_image_data(self, sr_num,
                               image_id, dtu_scan_dir, target_dir):
        
        
        out_sr_num = str(sr_num).zfill(4)
        
        # TODO: @ Implement
        K = None
        Rt = None
        size = None
        gravity = None
        image_data = np.ones(shape=tuple(np.ravel(size)))
        DUMMY_ROLL_PITCH = [0, 0]
        
        
        # paths
        out_img_path = os.path.join(target_dir, DIR_IMAGES, out_sr_num+".png")
        out_calib_path, out_pose_path, out_size_path, out_gravity_gt_path,\
            out_gravity_pred_path, out_rp_gt_path, out_rp_pred_path \
                = self.create_absolute_output_paths(target_dir, out_sr_num)
        
        
        # save to files
        matplotlib.image.imsave(out_img_path, image_data)
        np.savetxt(out_calib_path, K)
        np.savetxt(out_pose_path, Rt)
        np.savetxt(out_size_path, size)
        np.savetxt(out_gravity_gt_path, gravity)
        np.savetxt(out_gravity_pred_path, gravity)
        np.savetxt(out_rp_gt_path, DUMMY_ROLL_PITCH)
        np.savetxt(out_rp_pred_path, DUMMY_ROLL_PITCH)
    
    def create_absolute_output_paths(self, target_dir, out_sr_num):
        out_calib_path = os.path.join(target_dir, DIR_CALIBRATION,
                                      f"K_{out_sr_num}.txt")
        out_pose_path = os.path.join(target_dir, DIR_REL_POSE,
                                     f"{out_sr_num}.txt")
        out_size_path = os.path.join(target_dir, DIR_CALIBRATION, 
                                     f"image_size_{out_sr_num}.txt")
        out_gravity_gt_path = os.path.join(target_dir, DIR_GRAVITY_GT,
                                           f"{out_sr_num}.txt")
        out_gravity_pred_path = os.path.join(target_dir, DIR_GRAVITY_PRED,
                                           f"{out_sr_num}.txt")
        out_rp_gt_path = os.path.join(target_dir, DIR_ROLL_PITCH_GT,
                                           f"{out_sr_num}.txt")
        out_rp_pred_path = os.path.join(target_dir, DIR_ROLL_PITCH_PRED,
                                           f"{out_sr_num}.txt")
                                           
        return out_calib_path,out_pose_path,out_size_path,out_gravity_gt_path,\
            out_gravity_pred_path,out_rp_gt_path,out_rp_pred_path
    
    def init_visn_folder_structure(self, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for dir_name in ALL_VISN_DIRS:
            os.makedirs(os.path.join(target_dir, dir_name), exist_ok=True)
    
    def compute_gravity_from_pose(self, Rt):
        R = Rt[:, 0:3]
        
        g_ref = np.array([[0], [0], [-1]], dtype=np.float64)
        
        g_new = R @ g_ref
        
        g_new = np.ravel(g_new)
        
        return g_new
    
    
if __name__ == "__main__":
    datagen = CameraPairDataGenerator()
    datagen.get_all()
    
    