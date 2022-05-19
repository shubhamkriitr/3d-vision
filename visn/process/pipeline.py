from visn.process.components import (BasePreprocessor, PoseEstimationProcessor,
                          BenchmarkingProcessor)
from visn.data.loader import SequentialDataLoader, GroupedImagesDataset
from visn.utils import logger, get_timestamp_str
from visn.config import read_config
from typing import Dict
import pandas as pd
import numpy as np
import os

class BasePipeline:
    def __init__(self, config: Dict = {}, **kwargs) -> None:
        self._init_from_config(config)
        self.setup_pipeline_steps()
        self.output_dir = "pipeline_outputs" # TODO: read from config

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
            self.save_summary_csv
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
            step(outputs)
            
        return outputs
    
    def save_summary_csv(self, *args, **kwargs):
        outputs = args[0]
        
        error_summary = []
        item_names = [
            "R_err", "t_err", "min_iterations", 
            "max_epipolar_error", "success_prob",
            'refinements', 'iterations', 'num_inliers', 'inlier_ratio', 
            'model_score'
        ]
        
        solver_types = ["3pt_up",  "5pt"]
        columns = []
        for s in solver_types:
            for i in item_names:
                columns.append(s+"_"+i)
        
        data = {
            c: [] for c in columns
        }
        
        for batch in outputs:
            for sample in batch:
                R_err = sample["_stage_benchmark"]["pose_error_rotation"]
                t_err = sample["_stage_benchmark"]["pose_error_translation"]
                data["3pt_up_R_err"].append(R_err["3pt_up"])
                data["5pt_R_err"].append(R_err["5pt"])
                data["3pt_up_t_err"].append(t_err["3pt_up"])
                data["5pt_t_err"].append(t_err["5pt"])
                for s in ["3pt_up", "5pt"]:
                    d = sample["_stage_pose_estimate"]["ransac_options"]
                    for k in ["min_iterations", "max_epipolar_error", 
                            "success_prob"]:
                    
                        data[f"{s}_{k}"].append(d[s][k])
                for s in ["3pt_up", "5pt"]:
                    d = sample['_stage_pose_estimate']['solution_metadata']
                    for k in ['refinements', 'iterations', 'num_inliers', 
                              'inlier_ratio', 'model_score']:
                    
                        data[f"{s}_{k}"].append(d[s][k])
                
        
        output_stats_filename = f"{get_timestamp_str()}_run_stats.csv"
        
        
        filled_columns = [c for c in columns if len(data[c]) > 0]
        data_arr = [np.expand_dims(np.array(data[c]), axis=1) 
                    for c in filled_columns]
        data_arr = np.concatenate(data_arr, axis=1)
        
        df = pd.DataFrame(data_arr, columns=filled_columns)
        
        
        os.makedirs(self.output_dir, exist_ok=True)
        df.to_csv(os.path.join(self.output_dir, output_stats_filename),
                  index=True)
        
        


def extract_relevant_info(running_output):
    return running_output



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
        
        # slightly rotate the second image to test the effect
        from scipy.spatial.transform import Rotation
        Rdelta = Rotation.from_euler("zyx", [10, 0, 0], degrees=True)
        
        R1_new = Rdelta.as_matrix()@R1
        
        Rt1 = np.concatenate([R1_new, t1], axis=1)
        sample['input_relative_poses'][1] = Rt1
        
        
        
    def compute_dummy_gravity_vector_from_rotation_matrices(self):
        pass
        
        


if __name__ == "__main__":
    pass