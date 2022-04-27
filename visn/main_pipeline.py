from visn.data.synthetic import CameraPairDataGeneratorUpright3Pt
from visn.benchmark.timing import benchmark_runtime

import poselib


@benchmark_runtime
def estimate_relative_pose(*args, **kwargs):
    return poselib.estimate_relative_pose(*args, **kwargs)

@benchmark_runtime
def estimate_relative_pose_3pt_upright(*args, **kwargs):
    return poselib.estimate_relative_pose_3pt_upright(*args, **kwargs)

class PoseLibAdapter(object):
    def __init__(self) -> None:
        self.cache = None
    
    def solve_5pt(self, *args, **kwargs):
        x2d_1, x2d_2 = args
        info = poselib.relpose_5pt(x2d_1, x2d_2)
        info2 = poselib.essential_matrix_5pt(x2d_1, x2d_2)
        return info, info2

    def solve(self, *args, **kwargs):
        x2d_1, x2d_2 = self.to_list_of_nd_array(args[0][:, 0:2]), \
            self.to_list_of_nd_array(args[1][:, 0:2])

        # dummy params
        cam1 = {'model': 'SIMPLE_PINHOLE', 'width': 1200, 'height': 800, 'params': [4, 0, 0]}
        cam2 = {'model': 'SIMPLE_PINHOLE', 'width': 1200, 'height': 800, 'params': [4, 0, 0]}
        
        info = estimate_relative_pose(
                            x2d_1, x2d_2, cam1, cam2, poselib.RansacOptions(),
                            poselib.BundleOptions()
                            )
        # Use this: https://gitlab.ethz.ch/kumarsh/poselib-3dv/-/tree/visn
        # to build new poselib pybind
        info2  = estimate_relative_pose_3pt_upright(
                            x2d_1, x2d_2, cam1, cam2, poselib.RansacOptions(),
                            poselib.BundleOptions()
                            )
        _ = info2
        return info
    
    def solve_upright_3pt(self, *args):
        x2d_1, x2d_2 = args
        info = poselib.relpose_upright_3pt(x2d_1, x2d_2)
        return info
    
    def to_list_of_nd_array(self, x):
        x_ = [] # TODO: replace with an optimal version
        for i in range(x.shape[0]):
            x_.append(x[i])
        return x_

datagen = CameraPairDataGeneratorUpright3Pt()
x1, x2, rotation, translation, essential_matrix = datagen.get_all()
estimator = PoseLibAdapter()

estimator.solve(x1, x2)








