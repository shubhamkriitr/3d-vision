from visn.process.components import (BasePreprocessor, PoseEstimationProcessor,
                          BenchmarkingProcessor)
from visn.data.loader import SequentialDataLoader, GroupedImagesDataset
from visn.utils import logger
from visn.config import read_config
from typing import Dict


class BasePipeline:
    def __init__(self, config: Dict = {}, **kwargs) -> None:
        self._init_from_config(config)
        self.setup_pipeline_steps()

    def _init_from_config(self, config):
        # update config with default configuration
        self.config = {**read_config(), **config}

        # TODO get objects from factory
        self.dataset = GroupedImagesDataset(config=self.config["dataset"])
        self.dataloader = SequentialDataLoader(dataset=self.dataset,
                                               config=self.config["dataloader"])
        self.preprocessor = BasePreprocessor(self.config["preprocessor"])
        self.pose_estimation_processor = PoseEstimationProcessor()
        self.benchmarking_processor = BenchmarkingProcessor(pipeline=self)
    
    def setup_pipeline_steps(self):
        # TODO: use these steps
        # self.step_before_begin = [
            
        # ]
        self.steps = [
            # AdHocTransforms().process,
            self.preprocessor.process,
            self.pose_estimation_processor.process,
            self.benchmarking_processor.process
        ]
        self.steps_after_end = [
            self.print_full_summary
        ]
        
        
            
    def run(self):
        outputs = []
        for data in self.dataloader:
            running_output = data
            for step  in self.steps:
                running_output = step(running_output)
                
            output = extract_relevant_info(running_output)
            outputs.append(output)
            
        for step in self.steps_after_end:
            step()
            
        return outputs
    
    def print_full_summary(self, *args, **kwargs):
        logger.debug(f"TODO#") # TODO


def extract_relevant_info(running_output):
    return running_output  #TODO



class AdHocTransforms(BasePreprocessor):
    
    def __init__(self, config: Dict = ..., **kwargs) -> None:
        super().__init__(config, **kwargs)
    
    def _init_from_config(self, config):
        pass
        
    def process_one_sample(self, sample):
        x = 1
        y = x +1
        import numpy as np
        from copy import deepcopy
        common_chosen_axis = np.array([[0], [0], [-1]], dtype=np.float64)
        
        
        Rt0, Rt1 = sample['input_relative_poses']
        R0, t0 = Rt0[:, 0:3], Rt0[:, 3:4]
        R1, t1 = Rt1[:, 0:3], Rt1[:, 3:4]
        
        g0 = R0 @ common_chosen_axis
        g1 = R1 @ common_chosen_axis
        
        g0 = g0 / np.linalg.norm(g0)
        g1 = g1 / np.linalg.norm(g1)
        
        g0 = np.ravel(g0)
        g1 = np.ravel(g1)
        
        logger.warning(f"Overriding input_gravity")
        
        sample['input_gravity'] = [g0, g1]
        sample['input_roll_pitch_gt'] = deepcopy(sample['input_gravity'])
        sample['input_gravity_gt'] = deepcopy(sample['input_gravity'])
        
        
    def compute_dummy_gravity_vector_from_rotation_matrices(self):
        pass
        
        


if __name__ == "__main__":
    pass