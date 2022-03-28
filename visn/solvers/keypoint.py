
import numpy as np
import pycolmap
import cv2 as cv
from PIL import Image, ImageOps

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
    
    def extract_sift_features(self, image: np.ndarray):
        # https://docs.opencv.org/4.x/d0/d13/classcv_1_1Feature2D.html#a7fc43191552b06aa37fbe5522f5b0c71
        # second argument here is mask
        keypoints, descriptors = self.sift.detectAndCompute(image, None)

        img=cv.drawKeypoints(image, keypoints, np.copy(image), 
                        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite('sift_keypoints.jpg', img)
        return keypoints, descriptors
    
    def load_image(self, path_):
        img = cv.imread(path_)
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        return img





if __name__ == "__main__":
    kpm1 = KeyPointMatcher({})
    kpm2 = OpenCvKeypointMatcher({})
    def test_kpm(kpm):
        path1 = "res_/scene0000_00/rgb/224.png"
        path2 = "res_/scene0000_00/rgb/256.png"

        img1 = kpm.load_image(path1)
        img2 = kpm.load_image(path2)

        keypoints_1, descriptors_1 = kpm.extract_sift_features(img1)
        keypoints_2, descriptors_2 = kpm.extract_sift_features(img2)

        keypoints_1 = None
    
    test_kpm(kpm1)
    test_kpm(kpm2)