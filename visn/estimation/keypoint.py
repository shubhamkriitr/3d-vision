
from time import sleep
import numpy as np
# import pycolmap
import cv2 as cv
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

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
        path1 = './visn/examples/images/0006.png' # queryImage
        path2 = './visn/examples/images/0005.png' # trainImage

        img1 = kpm.load_image(path1)
        img2 = kpm.load_image(path2)

        keypoints_1, keypoints_2 = kpm.get_matches(img1,img2, 0.25)

        H, _ = cv.findHomography(keypoints_1, keypoints_2, cv.RANSAC,5.0)
        img1_warp = kpm.warp_image(img1, H)
        result = np.concatenate((img1_warp,img2), axis=1)
        
        plt.imshow(result, 'gray')
        plt.show()
    
    # test_kpm(kpm1)
    test_kpm(kpm2)
