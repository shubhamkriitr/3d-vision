from visn.process import (BasePreprocessor, PoseEstimationProcessor,
                          BenchmarkingProcessor)
from visn.data.loader import SequentialDataLoader, GroupedImagesDataset
from visn.utils import logger
class BasePipeline:
    def __init__(self, config = None, **kwargs) -> None:
        self._init_from_config(config)
        
        self.setup_pipeline_steps()
        
    
    def _init_from_config(self, config):
        if config is None:
            self.config = {
               "input_loader": {
                   "dataset": "GroupedImagesDataset",
                   "loader": "SequentialDataLoader"
               },
               "preprocessor": "BasePreprocessor",
               "benchmarker": None
            }
        else:
            self.config = config
        # TODO get objects from factory
        self.dataset = GroupedImagesDataset()
        self.dataloader = SequentialDataLoader(dataset=self.dataset,
                                               batch_size=1)
        self.preprocessor = BasePreprocessor()
        self.pose_estimation_processor = PoseEstimationProcessor()
        self.benchmarking_processor = BenchmarkingProcessor(pipeline=self)
        
        
    
    def setup_pipeline_steps(self):
        # TODO: use these steps
        # self.step_before_begin = [
            
        # ]
        self.steps = [
            self.preprocessor.process,
            self.pose_estimation_processor.process,
            self.benchmarking_processor.process
        ]
        self.steps_after_end = [
            self.print_summary
        ]
        
        
            
    def run(self):
        outputs = []
        for data in self.dataloader:
            running_output = data
            for step  in self.steps:
                running_output = step(running_output)
                
        for step in self.steps_after_end:
            step()
            
        return outputs
    
    def print_full_summary(self, *args, **kwargs):
        logger.debug(f"TODO#") # TODO



        
        
        
        


if __name__ == "__main__":
    pass