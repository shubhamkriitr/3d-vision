import numpy as np
from scipy.spatial.transform import Rotation


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
    
if __name__ == "__main__":
    datagen = CameraPairDataGenerator()
    datagen.get_all()
    
    