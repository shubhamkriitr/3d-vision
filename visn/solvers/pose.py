import poselib
import os
import numpy as np
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
    
    def to_list_of_nd_array(self, x):
        x_ = [] # TODO: replace with an optimal version
        for i in range(x.shape[0]):
            x_.append(x[i])
        return x_

def Triangulation(x1,x2, rot, t):
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

    x1_3d,x2_3d = Triangulation(x1,x2,rotation,translation)

    #for pose in ans1[0]:
    #    x1_3d,x2_3d = Triangulation(x1,x2,pose.R,pose.t)
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
    #    x1_3d,x2_3d = Triangulation(x1,x2,pose.R,pose.t)
    #    n_positive = (x1_3d[:,2]>0) & (x2_3d[:,2]>0)
    #    print(n_positive.sum())
    
    x1, x2, rotation, translation, essential_matrix = datagen.get_all(
        num_samples=None)
    
    ans4 = estimator.solve_5pt_ransac(x1, x2)
    
    return [ans1, ans2, ans3, ans4]

if __name__ == "__main__":
    test()
    
        