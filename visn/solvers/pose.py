import poselib
import os
print(os.getcwd())
from visn.data.synthetic import CameraPairDataGenerator


default_ransac_options = poselib.RansacOptions()
default_bundle_options = poselib.BundleOptions()

# essential_matrix_5pt
# relpose_5pt
class PoseLibAdapter(object):
    def __init__(self) -> None:
        self.cache = None
    
    def solve_5pt(self, *args, **kwargs):
        x2d_1, x2d_2 = args
        info = poselib.relpose_5pt(x2d_1, x2d_2)
        info2 = poselib.essential_matrix_5pt(x2d_1, x2d_2)
        return info, info2

    def solve_5pt_ransac(self, *args, **kwargs):
        x2d_1, x2d_2 = self.to_list_of_nd_array(args[0][:, 0:2]), \
            self.to_list_of_nd_array(args[1][:, 0:2])

        # dummy params
        cam1 = {'model': 'SIMPLE_PINHOLE', 'width': 1200, 'height': 800, 'params': [1, 1, 1]}
        cam2 = {'model': 'SIMPLE_PINHOLE', 'width': 1200, 'height': 800, 'params': [1, 1, 1]}
        
        info = poselib.estimate_relative_pose(
                            x2d_1, x2d_2, cam1, cam2, poselib.RansacOptions(),
                            poselib.BundleOptions()
                            )
        return info
    
    def to_list_of_nd_array(self, x):
        x_ = [] # TODO: replace with an optimal version
        for i in range(x.shape[0]):
            x_.append(x[i])
        return x_


def test():
    datagen = CameraPairDataGenerator()
    estimator = PoseLibAdapter()
    x1, x2, rotation, translation, essential_matrix = datagen.get_all(
        num_samples=5)
    # x1, x2 = x1[0:4], x2[0:4]
    ans1 = estimator.solve_5pt(x1, x2)
    
    x1, x2, rotation, translation, essential_matrix = datagen.get_all(
        num_samples=5)
    
    # x1, x2 = x1[0:4], x2[0:4]
    ans2 = estimator.solve_5pt_ransac(x1, x2)
    
    x1, x2, rotation, translation, essential_matrix = datagen.get_all(
        num_samples=None)
    
    ans3 = estimator.solve_5pt(x1, x2)
    
    x1, x2, rotation, translation, essential_matrix = datagen.get_all(
        num_samples=None)
    
    ans4 = estimator.solve_5pt_ransac(x1, x2)
    
    return [ans1, ans2, ans3, ans4]

if __name__ == "__main__":
    test()
    
        