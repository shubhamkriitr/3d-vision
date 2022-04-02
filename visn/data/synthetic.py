import numpy as np
from scipy.spatial.transform import Rotation


class SyntheticDataGenerator(object):
    def __init__(self) -> None:
        pass
    

class CameraPairDataGenerator(SyntheticDataGenerator):
    def __init__(self) -> None:
        super(CameraPairDataGenerator, self).__init__()
    
    def get_all(self):
        rotation = Rotation.from_euler('zyx', [90, 0, 0], degrees=True)
        rotation = rotation.as_matrix()
        translation = np.array([[1], [2], [1]], dtype=np.float32)
        t_cross = self.get_a_cross(translation)
        essential_matrix = t_cross @ rotation
        return rotation, translation, essential_matrix
    
    

    
    def get_a_cross(self, a):
        a = np.ravel(a)
        if a.shape[0] != 3:
            raise AssertionError(f"`a` must be a 3 vector")
        return np.array([[0, -a[2], a[1]],
                         [a[2], 0, -a[0]],
                         [-a[1], a[0], 0]], dtype=np.float32)
        


if __name__ == "__main__":
    datagen = CameraPairDataGenerator()
    datagen.get_all()
    
    