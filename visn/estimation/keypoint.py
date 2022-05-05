from time import sleep
import numpy as np
# import pycolmap
import cv2 as cv
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from typing import List

pycolmap = None # FIXME
# Adapters to use pycolmap/opencv for keypoints
class KeyPointMatcher(object):
    def __init__(self, config):
        pass
    
    def extract_sift_features(self, image: np.ndarray):
        # not using scores (the 2nd return value)
        keypoints, _, descriptors = pycolmap.extract_sift(image)

        return keypoints, descriptors


    def load_image(self, path_):
        img = Image.open(path_)
        img = img.convert('RGB')
        img = ImageOps.grayscale(img)
        img = np.array(img).astype(np.float) / 255.
        return img
    
class OpenCvKeypointMatcher(KeyPointMatcher):

    def __init__(self, config):
        super().__init__(config)
        self.sift = cv.SIFT_create()
        self.bf = cv.BFMatcher()
    
    def extract_sift_features(self, image: np.ndarray):
        # https://docs.opencv.org/4.x/d0/d13/classcv_1_1Feature2D.html#a7fc43191552b06aa37fbe5522f5b0c71
        # second argument here is mask
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def get_matches(self, img1, img2, ratio):
        kp1, des1 = self.extract_sift_features(img1)
        kp2, des2 = self.extract_sift_features(img2)
        matches = self.bf.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < ratio*n.distance:
                good.append(m)
        kp1_loc = np.array([kp1[gd.queryIdx].pt for gd in good])
        kp2_loc = np.array([kp2[gd.trainIdx].pt for gd in good])
        return kp1_loc,kp2_loc
    
    def warp_image(self, img, H, k):
        k_inv = np.linalg.inv(k)
        img_warp_matrix = k@H@k_inv
        return cv.warpPerspective(img, img_warp_matrix, (img.shape[1], img.shape[0]))

    def warp_feature(self, feat, H):
        # feat (n*3), H (3*3)
        return feat@H.T

    def load_image(self, path_):
        img = cv.imread(path_)
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        return img


class KeypointMatchBenchmarker:
    def __init__(self, relative_pose: List[List[float]], k: List[List[float]]):
        # FIXME we got one or multiple bugs
        # TODO Nico
        # This can be used for the pipeline so far as a pseudo-Benchmarker
        self.rel_pose_0 = np.array(relative_pose[0])
        self.rel_pose_1 = np.array(relative_pose[1])
        self.k = np.array(k)
        self.k_inv = np.linalg.inv(self.k)

        # compute relative pose from image 0 to image 1
        self.rel_pose_0_to_1 = self.rel_pose_a_to_b(self.rel_pose_0[:, :3], self.rel_pose_0[:, 3],
                                                    self.rel_pose_1[:, :3], self.rel_pose_1[:, 3])

        # compute fundamental and essential matrices
        self.rot_0_to_1 = self.rel_pose_0_to_1[:, :3]
        self.trans_0_to_1 = self.rel_pose_0_to_1[:, 3]
        self.trans_0_to_1_cross = np.array([[0, -self.trans_0_to_1[2], self.trans_0_to_1[1]],
                                            [self.trans_0_to_1[2], 0, self.trans_0_to_1[0]],
                                            [-self.trans_0_to_1[1], self.trans_0_to_1[0], 0]])
        self.F = self.rot_0_to_1 @ self.trans_0_to_1_cross
        self.E = self.k_inv.T @ self.F @ self.k_inv
        self.E = self.E * np.linalg.norm(self.E)  # normalize with respect to frobenius norm

    @staticmethod
    def rel_pose_a_to_b(pose_a, trans_a, pose_b, trans_b):
        trans_a_to_b = pose_a.T @ (trans_b - trans_a).reshape((3, 1))
        pose_a_to_b = pose_a.T @ pose_b
        rel_pose = np.concatenate((pose_a_to_b, trans_a_to_b), axis=1)
        return rel_pose

    def check(self, kpt_0_: List[List[float]], kpt_1_: List[List[float]], epsilon: float):
        """
        Gives us a geometric error for each of the keypoints by computing
        the shortest distance between kpt_1 and the epipolar line in image 1.
            Input:
                kpt_0_: list of normalized keypoints of image 0 (matching order of kpt_1_)
                kpt_1_: list of normalized keypoints of image 1 (matching order of kpt_0_)
        """
        kpt_0_, kpt_1_ = np.array(kpt_0_), np.array(kpt_1_)  # convert to numpy array
        num_kpts = kpt_0_.shape[0]
        ones = np.ones((num_kpts, 1))
        kpt_0_, kpt_1_ = np.hstack((kpt_0_, ones)), np.hstack((kpt_1_, ones))
        r = []
        for ind, (kpt_0, kpt_1) in enumerate(zip(kpt_0_, kpt_1_)):
            kpt_0 = kpt_0.reshape((3, 1))
            kpt_1 = kpt_1.reshape((3, 1))
            min_distance_to_epipolar_plane = kpt_1.T @ self.E @ kpt_0
            r.append(abs(min_distance_to_epipolar_plane.item(0)))
        # TODO normalize (see paper: Fast Iterative Five point Relative Pose Estimation)

        # return outlier or inlier bool vector
        inliers = np.array(r) < epsilon
        score = np.sum(inliers) / num_kpts
        return inliers, score


def trial():
    img1 = cv.imread('./visn/examples/images/0006.png') # queryImage
    img2 = cv.imread('./visn/examples/images/0005.png') # trainImage
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()
    sleep(60)



if __name__ == "__main__":
    # kpm1 = KeyPointMatcher({})
    kpm2 = OpenCvKeypointMatcher({})
    def test_kpm(kpm):
        #path1 = "res_/scene0000_00/rgb/224.png"
        #path2 = "res_/scene0000_00/rgb/256.png"
        path1 = '../../visn/examples/images/0006.png' # queryImage
        path2 = '../../visn/examples/images/0005.png' # trainImage
        k_path = '../../visn/examples/images/K.txt' # K matrix

        img1 = kpm.load_image(path1)
        img2 = kpm.load_image(path2)
        with open(k_path, "r") as f:
            content = f.read()
            k = [[float(y) for y in x.split(" ")] for x in content.split("\n") if x]

        keypoints_1, keypoints_2 = kpm.get_matches(img1,img2, 0.25)
        print(keypoints_1)

        H, _ = cv.findHomography(keypoints_1, keypoints_2, cv.RANSAC,5.0)
        img1_warp = kpm.warp_image(img1, H, k)
        result = np.concatenate((img1_warp,img2), axis=1)
        
        plt.imshow(result, 'gray')
        plt.show()
    
    # test_kpm(kpm1)
    test_kpm(kpm2)
