import numpy as np
from scipy.spatial.transform import Rotation
import os
import shutil
import matplotlib
import imageio
from visn.utils import logger


class SyntheticDataGenerator(object):
    def __init__(self) -> None:
        pass
    

class CameraPairDataGenerator(SyntheticDataGenerator):
    """ Used for generating synthetic data for test cases. This approach allows
        for absolute knowledge of keypoint matches as well as relative pose.
        This feature is used to test the pipeline and helps with debugging.
    """
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


# TODO: Move to constants
DIR_CALIBRATION = "calibration"
DIR_GRAVITY_GT = "gravity_gt"
DIR_GRAVITY_PRED = "gravity_pred"
DIR_IMAGES = "images"
DIR_REL_POSE = "relative_pose"
DIR_ROLL_PITCH_GT = "roll_pitch_gt"
DIR_ROLL_PITCH_PRED = "roll_pitch_pred"
DIR_MATCHED_KEY_POINTS = "matched_keypoints"

ALL_VISN_DIRS = [
    DIR_CALIBRATION,
    DIR_GRAVITY_GT,
    DIR_GRAVITY_PRED,
    DIR_IMAGES,
    DIR_REL_POSE,
    DIR_ROLL_PITCH_GT,
    DIR_ROLL_PITCH_PRED,
    DIR_MATCHED_KEY_POINTS
]


class SyntheticVisnDataGenerator(object):
    """ Used for generating synthetic data for test cases. This approach allows
        for absolute knowledge of keypoint matches as well as relative pose.
        This feature is used to test the pipeline and helps with debugging.
    """
    def __init__(self) -> None:
        pass
    
    def generate(self, num_pairs, output_dir, static=False,
                 *args, **kwargs):
        self.init_visn_folder_structure(output_dir)
        
        assert num_pairs == 1, "currently only single pair (1) is supported"
        K0, K1, Rt0, Rt1, world_points = self.get_seed_pair_data(static=static)
        
        kp_1 = self.transform_world_points_to_img(world_points, K0, Rt0)
        kp_2 = self.transform_world_points_to_img(world_points, K1, Rt1)
        
        g_0 = self.compute_gravity_from_pose(Rt0)
        g_1 = self.compute_gravity_from_pose(Rt1)
        
        self.save_single_image_data(
            sr_num=1, target_dir=output_dir,
            K=K0, Rt=Rt0, size=[[500], [500]], 
            gravity=g_0
        )
        self.save_single_image_data(
            sr_num=2, target_dir=output_dir,
            K=K1, Rt=Rt1, size=[[500], [500]], 
            gravity=g_1
        )
        
        self.save_matched_keypoints(
            target_dir=output_dir, sr_num_1=1, sr_num_2=2, kp_1=kp_1, kp_2=kp_2
        )
        
        groups_path = os.path.join(output_dir, "groups.txt")
        self.save_group(sr_num_1=1, sr_num_2=2, output_path=groups_path)
        
    def save_group(self, sr_num_1, sr_num_2, output_path, mode="overwrite"):
        """
        TODO: add mode `append` later to modify same group file
        """
        out_sr_num_1 = str(sr_num_1).zfill(4)
        out_sr_num_2 = str(sr_num_2).zfill(4)
        assert mode == "overwrite" # TODO remove later
        with open(output_path, "w") as f:
            f.write(f"{out_sr_num_1} {out_sr_num_2}\n")

    def save_matched_keypoints(self, target_dir, 
                               sr_num_1, sr_num_2, kp_1, kp_2):
        
        matched_kps = np.concatenate([kp_1, kp_2], axis=1)
        out_sr_num_1 = str(sr_num_1).zfill(4)
        out_sr_num_2 = str(sr_num_2).zfill(4)
        
        filename = f"kp_matches_{out_sr_num_1}_{out_sr_num_2}.txt"
        
        kp_match_output_path = os.path.join(target_dir, DIR_MATCHED_KEY_POINTS,
                                            filename)
        
        np.savetxt(kp_match_output_path, matched_kps)    
        
    
    def transform_world_points_to_img(self, xW, K, Rt):
        """ 
        xW of shape (N, 3)
        """
        xW_hom = self.to_homogeneous(xW)
        x_hom = K @ Rt @ xW_hom.T # 3 x 4  x   4 x N
        x_hom = x_hom.T # N x 3
        if np.any(x_hom[:, 2] < 0):
            logger.warning(f"Some points are not in front of the camera")
        x = self.to_non_homogeneous(x_hom)
        return x
    
    def to_homogeneous(self, x):
        """ 
        x of shape (N, m) 
        returns array of shape (N, m+1)
        """
        ones = np.ones(shape=(x.shape[0], 1), dtype=x.dtype)
        x_hom = np.concatenate([x, ones], axis=1)
        return x_hom
    
    def to_non_homogeneous(self, x):
        """ 
        x of shape (N, m) 
        returns array of shape (N, m-1)
        """
        x_non_hom = x / x[:, x.shape[1]-1:x.shape[1]]
        x_non_hom = x_non_hom[:, 0:-1] #drop last col
        return x_non_hom
        

    def get_seed_pair_data(self, static=True):
        K0 = np.array(
            [
                [100, 0, 5],
                [0, 100, 5],
                [0, 0, 1]
            ]
        )
        K1 = np.array(
            [
                [100, 0, 10],
                [0, 100, 10],
                [0, 0, 1]
            ]
        )
        
        R0 = Rotation.from_euler("zyx", [90, 5 , 5], degrees=True).as_matrix()
        R1 = Rotation.from_euler("zyx", [20, 10 , 10], degrees=True).as_matrix()
        
        c0 =  np.array([[10], [5], [1]], dtype=np.float64)
        c1 =  np.array([[20], [10], [5]], dtype=np.float64)
        
        t0 = - R0 @ c0
        t1 = - R1 @ c1
        
        Rt0 = np.concatenate([R0, t0], axis=1)
        Rt1 = np.concatenate([R1, t1], axis=1)
        
        
        
        
        world_points = np.array(
            [[10, 30, 10],
            [10, 40, 10],
            [20, 20, 15],
            [20, 30, 15],
            [15, 12, 20],
            [15, 0, 20],
            [15, 15, 25],
            [15, 25, 25],
            [0, 30, 30],
            [30, 0, 30],
            [35, 35, 35],
            [40, 35, 35],
            [12, 10, 80],
            [12, 15, 80],
            [12, 19, 80],
            [19, 19, 80],
            [19, 19, 90],
            [19, 19, 95],
            [100, 100, 100]], dtype=np.float64)
        
        if static:
            return K0, K1, Rt0, Rt1, world_points
        
        
        x_range = [0, 50]
        y_range = [0, 100]
        z_range = [100, 200]
        
        logger.info(f"Not using hard coded data for now. "
                    f"Using random data points from"
                    f" this grid -> x: {x_range} , y: {y_range}, z: {z_range}"
                    f" To use static data; set `static` arg")
        world_points = self.get_random_world_points_from_grid(
            n_samples=250, x_range=x_range, y_range=y_range, z_range=z_range
        )
        
        
        
        return K0, K1, Rt0, Rt1, world_points
    
    def get_random_world_points_from_grid(self, n_samples,
                                        x_range, y_range, z_range):
        size = (n_samples, 1)
        x = np.random.uniform(low=x_range[0], high=x_range[1], size=size)
        y = np.random.uniform(low=y_range[0], high=y_range[1], size=size)
        z = np.random.uniform(low=z_range[0], high=z_range[1], size=size)
        
        world_points = np.concatenate([x, y, z], axis=1)
        
        return world_points
        
    
        
    
    def save_single_image_data(self, sr_num, target_dir, K,
                               Rt, size, gravity, *args):
        
        
        out_sr_num = str(sr_num).zfill(4)
        
        # TODO: @ Implement
        image_data = np.ones(shape=tuple(np.ravel(size)))
        DUMMY_ROLL_PITCH = [0, 0]
        
        
        # paths
        out_img_path = os.path.join(target_dir, DIR_IMAGES, out_sr_num+".png")
        out_calib_path, out_pose_path, out_size_path, out_gravity_gt_path,\
            out_gravity_pred_path, out_rp_gt_path, out_rp_pred_path \
                = self.create_absolute_output_paths(target_dir, out_sr_num)
        
        
        # save to files
        imageio.imwrite(out_img_path, image_data)
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
    # datagen = CameraPairDataGenerator()
    # datagen.get_all()
    datagen = SyntheticVisnDataGenerator()
    datagen.generate(1, "__temp__/synthetic_data", static=False)
    
    
    