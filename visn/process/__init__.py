from visn.solvers.keypoint import OpenCvKeypointMatcher
import numpy as np

class BasePreprocessor(object):
    """This preprocessor works on image pairs. For more than 2 images in 
    a sample, only the first two will be used.
    """
    def __init__(self, config = None, **kwargs) -> None:
        self._init_from_config(config)
        self.pipeline = None # to be used later (for shared context etc.)
        
    
    def _init_from_config(self, config):
        if config is None:
            self.config = {
                "keypoint_matcher": "OpenCvKeypointMatcher" # use factory: TODO
            }
        else:
            self.config = config
            
        assert self.config["keypoint_matcher"] == "OpenCvKeypointMatcher"
        self.kpm = OpenCvKeypointMatcher()
        self.keypoint_threshold = 0.25
            
    def process(self, batch_):
        # Assumes batch_ is a list of dictionaries 
        # {"input": [img_1, img_2, ...],
        # "K": [K_0, K_1, ....]}
        
        output = []
        
        for sample in batch_:
            k_inverse = [np.linalg.inv(k) for k in sample["K"]]
            input_images = sample["input_images"]
            
            keypoints_0, keypoints_1 = self.kpm.get_matches(
                input_images[0], input_images[1], self.keypoint_threshold)
            
            sample["stage.preprocess"] = {
                "K_inverse": k_inverse,
                "keypoints": [keypoints_0, keypoints_1],
                "normalized_keypoints": None ,
            }

    def normalized_keypoints(self, intrinsic_matrix : np.ndarray,
                             keypoints: np.ndarray):
        pass
        
