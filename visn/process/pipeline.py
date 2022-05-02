from visn.process import BasePreprocessor, PoseEstimationProcessor
from visn.data.loader import SequentialDataLoader, GroupedImagesDataset
class BasePipeline:
    def __init__(self, config = None, **kwargs) -> None:
        self._init_from_config(config)
        
    
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
        
            
    def run(self):
        outputs = []
        for data in self.dataloader:
            out_ = self.preprocessor.process(data)
            out_ = self.pose_estimation_processor.process(out_)
            outputs.append(out_)
        return outputs



        
        
        
        


if __name__ == "__main__":
    pass