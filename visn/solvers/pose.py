import poselib
import os
import numpy as np
print(os.getcwd())
from visn.data.synthetic import (CameraPairDataGenerator,
    CameraPairDataGeneratorUpright3Pt)
from visn.examples.fetch_example import get_matched_kps, get_k_matrix
from visn.solvers.keypoint import OpenCvKeypointMatcher
from visn.utils import logger
import matplotlib.pyplot as plt


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
        cam1 = {'model': 'SIMPLE_PINHOLE', 'width': 1200, 'height': 800, 'params': [4, 0, 0]}
        cam2 = {'model': 'SIMPLE_PINHOLE', 'width': 1200, 'height': 800, 'params': [4, 0, 0]}
        
        info = poselib.estimate_relative_pose(
                            x2d_1, x2d_2, cam1, cam2, poselib.RansacOptions(),
                            poselib.BundleOptions()
                            )
        # Use this: https://gitlab.ethz.ch/kumarsh/poselib-3dv/-/tree/visn
        # to build new poselib pybind
        info2  = poselib.estimate_relative_pose_3pt_upright(
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

def triangulate(x1,x2, rot, t):
    if t.ndim == 1:
        t = t[:,np.newaxis]
    p = np.concatenate([rot, t], axis=1)
    # x1, x1 shape [n,3]
    ones = np.ones([x1.shape[0],1,1])
    zeros = np.zeros([x1.shape[0],1,1])

    # plane (l1x,l1y,l2x,l2y) [n,1,4]
    l1x = np.concatenate([-ones, zeros, x1[:,0,np.newaxis,np.newaxis],zeros], axis=2)
    l1y = np.concatenate([zeros, -ones, x1[:,1,np.newaxis,np.newaxis],zeros], axis=2)
    l2x = np.concatenate([-ones, zeros, x2[:,0,np.newaxis,np.newaxis]], axis=2) @ p
    l2y = np.concatenate([zeros, -ones, x2[:,1,np.newaxis,np.newaxis]], axis=2) @ p
    
    # plane (l1x,l1y,l2x,l2y) [n,4,4]
    planes = np.concatenate([l1x,l2x,l1y,l2y],axis=1)
    # u = (n, 4, 4), s=(n, 4), vh=(n, 4, 4)
    u, s, vh = np.linalg.svd(planes)
    x1_3d = vh[:,3,:]
    x1_3d = x1_3d / x1_3d[:,3,np.newaxis]
    #check =  planes @ ans[..., np.newaxis]
    # return (N,4) 
    x2_3d = (p @ x1_3d.T).T
    x2_3d = np.concatenate([x2_3d,ones[:,:,0]], axis=1)
    return x1_3d, x2_3d





def test():
    datagen = CameraPairDataGenerator()
    estimator = PoseLibAdapter()
    x1, x2, rotation, translation, essential_matrix = datagen.get_all(
        num_samples=5)
    # x1, x2 = x1[0:4], x2[0:4]
    ans1 = estimator.solve_5pt(x1, x2)

    x1_3d,x2_3d = triangulate(x1,x2,rotation,translation)

    #for pose in ans1[0]:
    #    x1_3d,x2_3d = triangulate(x1,x2,pose.R,pose.t)
    #    n_positive = (x1_3d[:,2]>0) & (x2_3d[:,2]>0)
    #    print(pose.R)
    #    print(pose.t)
    #    print(x1_3d)
    #    print(x2_3d)
    #    print(n_positive.sum())

    x1, x2, rotation, translation, essential_matrix = datagen.get_all(
        num_samples=5)
    
    # x1, x2 = x1[0:4], x2[0:4]
    ans2 = estimator.solve_5pt_ransac(x1, x2)
    
    x1, x2, rotation, translation, essential_matrix = datagen.get_all(
        num_samples=None)
    
    ans3 = estimator.solve_5pt(x1, x2)

    #for pose in ans3[0]:
    #    x1_3d,x2_3d = triangulate(x1,x2,pose.R,pose.t)
    #    n_positive = (x1_3d[:,2]>0) & (x2_3d[:,2]>0)
    #    print(n_positive.sum())
    
    x1, x2, rotation, translation, essential_matrix = datagen.get_all(
        num_samples=None)
    
    ans4 = estimator.solve_5pt_ransac(x1, x2)
    
    return [ans1, ans2, ans3, ans4]


class TestUpright3PtSolver():
    def test_upright_3pt_solver(self):
        estimator = PoseLibAdapter()
        datagen = CameraPairDataGeneratorUpright3Pt()
        x1, x2, rotation, translation, essential_matrix = get_matched_kps(selected_images = [6,5]) #datagen.get_all(num_samples=3)
        
        #rotation = np.float64([[1,0,0],[0,1,0],[0,0,1]])
        #rotation = np.float64([[0.995,-0.09983,0],[0.09983,0.995,0],[0,0,1]])
        #rotation = np.float64(rotation)
        kpm = OpenCvKeypointMatcher({})
        
        # check image warp function
        path1 = './visn/examples/images/0006.png' # queryImage
        path2 = './visn/examples/images/0005.png' # trainImage
        img1 = kpm.load_image(path1)
        img2 = kpm.load_image(path2)
        k = get_k_matrix()
        img1_warp = kpm.warp_image(img1, rotation, k)
        result = np.concatenate((img1_warp,img2), axis=1)
        keypoints_1, keypoints_2 = kpm.get_matches(img1_warp,img2, 0.25) #you can use this point instead of x1,x2
        plt.imshow(result, 'gray')
        plt.show()

        # check feature warp
        x1 = kpm.warp_feature(x1, rotation)

        solutions = estimator.solve_upright_3pt(x1, x2)
        solution_comparison_info = self.validate_upright_3pt_solution(
            solutions, rotation, translation, essential_matrix
        )
        return {
            "solutions": solutions,
            "solution_comparison_info": solution_comparison_info,
            "groundtruth": {
                "rotation": rotation,
                "translation": translation,
                "essential_matrix": essential_matrix,
                "x1": x1,
                "x2": x2
            }
        }
    
    
    def validate_upright_3pt_solution(self, solutions, rotation, translation,
                                      essential_matrix):
        logger.info(f"Number of solutions: {len(solutions)}")
        solution_comparison_info = []
        for idx, solution_pose in enumerate(solutions):
            estimated_t = solution_pose.t
            estimated_rotation = solution_pose.R
            t_matched = self.check_translation(estimated_t, translation)
            r_matched = self.check_rotation(estimated_rotation, rotation)
            
            logger.info(f"#[{idx}][`t` matched: {t_matched}]"
                        f" ; [`R` matched: {r_matched}]")
            solution_comparison_info.append(
                {
                    "t_matched": t_matched,
                    "r_matched": r_matched
                }
            )
        
        return solution_comparison_info
            
    
    def check_translation(self, estimated_t, actual_t):
        actual_t = np.ravel(actual_t)
        actual_t = actual_t/actual_t[-1]
        estimated_t = estimated_t/estimated_t[-1]
        
        return np.allclose(estimated_t, actual_t)
    
    def check_rotation(self, estimated_rotation, actual_rotation):
        assert np.allclose(abs(np.linalg.det(estimated_rotation)), 1.0)
        assert np.allclose(abs(np.linalg.det(actual_rotation)), 1.0)
        return np.allclose(estimated_rotation, actual_rotation)
        
        
        
        
        
if __name__ == "__main__":
    test()
    tester = TestUpright3PtSolver()
    results = tester.test_upright_3pt_solver()
    print(results)
    
        