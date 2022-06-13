import numpy as np
from typing import List, Dict, Union
from types import SimpleNamespace
# import torch
torch = SimpleNamespace(**{"Tensor": str}) # for typing #FIXME


class BaseGravityEstimator(object):
    """ Estimate gravity vector used for 3-point solver in pose estimation"""
    def __init__(self, config=None, **kwargs) -> None:
        pass
    
    def _init_from_config(self, config):
        if config is None:
            self.config = {
            }
        else:
            self.config = config 
    
    def estimate_gravity(self, images: Union[np.ndarray, torch.Tensor]):
        """Estimate gravity vector in the global image coordinates.

        Args:
            `images` : Batch of color images
            
        Returns:
            gravity vectors in 3D non-homoeneous coordinates
        """
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        batch_size = images.shape[0]
        
        return np.ones(shape=(batch_size, 3))
            
        
        