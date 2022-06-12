# Estimating Relative Pose by Estimating Gravity

## Acknowledgement

We expect everything to work on an isolated python environment created 
as per the instructions below, but in case you face any issues running
the code please feel free to contact us (burgern@ethz.ch, kumarsh@ethz.ch,
timoscho@ethz.ch, and ywibowo@ethz.ch).

We have tested our code in an environment with the following specifications:
- Machine:
    - CPU: `11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz`
        - `x86_64 ` 
    - RAM: 16 GB
- OS: `Ubuntu 20.04.4 LTS`
- Python Version: `3.9.5`

Besides this, UprightNet model training was done on a node with GPU (`NVIDIATITANRTX`).


## Overview

The project is composed of 3 main components.
- UprightNet (for gravity estimation)
- PoseLib (contains C++ implementation of 5-point and 3-point estimators)
- visn (contains pipeline which benchmarks the two estimators using "python binding of PoseLib" and gravity estimates from "UprightNet")

For better modularity we track the above three components as separate projects.

### Steps to reproduce the results
The following is a quick overview of the steps to reproduce the results. Please refer to respective sections for details.
- Building `PoseLib` project and creating `poselib` python package
  - Build environment setups
- Using UprightNet to get the gravity estimates (TODO)
  - Here we assume UprightNet is already trained (for training steps refer section (TODO))
- ScanNet data preprocessing
  - (TODO)
  - (TODO)
- Executing the pipeline
  - `visn/main.py` is the entry point for running the pipeline
### Directory Structure

(TODO)
---
## Building Poselib Python Library

We have added a 3-point estimator class to the existing PoseLib library, which is essentially a class that combines already implemented 3-point solver, RANSAC and bundle adjustment. The final wheel file will be generated once you execute the following steps. (However we have also attached the final wheel file located at `artifacts/poselib-2.0.0-cp39-cp39-linux_x86_64.whl` so you can quickstart by skipping the following build steps)

> Steps to build

- Go inside `poselib-3dv` directory
- Extract `../artifacts/eigen-3.4.0.zip` to the current folder (Note: `eigen-3.4.0.zip` was downloaded from https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip )
- Make sure your default python version is `3.9.5`
  - Install `pybind11` in your python environment
    - `pip install pybind11`
- Copy the `Eigen` folder (which is inside extracted `eigen-3.4.0` directory) to `/usr/local/include/`
  - `sudo cp -r eigen-3.4.0/Eigen/ /usr/local/include/`
  - Run the following commands
    ```sh
    mkdir _build
    cmake -S . -B _build/ -DPYTHON_PACKAGE=ON -DWITH_BENCHMARK=ON -DCMAKE_INSTALL_PREFIX=_install
    cmake --build _build/ --target install -j 8
    cmake --build _build/ --target pip-package
    cmake --build _build/ --target install-pip-package
    ```
  - If all the above steps succeed, the python `poselib` package will be generated at the following location
    - `_build/pybind/pip_package/`
    - _e.g._ `_build/pybind/pip_package/poselib-2.0.0-cp39-cp39-linux_x86_64.whl`
  - The generated package should automatically be installed in the active python environment
  - To manually install the generated wheel to your target python environment
    - run _e.g._ `pip install _build/pybind/pip_package/poselib-2.0.0-cp39-cp39-linux_x86_64.whl`




---
## Running UprightNet

- Download the pretrained weights from [here](https://drive.google.com/file/d/15ZIFwPHP9W50YnsM4JPQGrlcvOeM3fM4/view?usp=sharing): https://drive.google.com/file/d/15ZIFwPHP9W50YnsM4JPQGrlcvOeM3fM4/view?usp=sharing
### Predicting Gravity

1. Download and extract from https://drive.google.com/drive/folders/1WdNAESqDYcUPQyXAW6PvlcdQIYlOEXIw:
	checkpoints.zip (pretrained weights)
	
2. Copy checkpoints folder to root directory of UprightNet repo

   If your computer does not have an gpu, check out the branch "no-gpu".

### Processing Scannet Data

### Predicting Gravity

1. Download and extract from https://drive.google.com/drive/folders/1V2KIsXIZ-2-5kGDaErTIpRNnBV2zhVjG?usp=sharing:
	checkpoints.zip (pretrained weights)
	
2. Copy checkpoints folder to root directory of UprightNet repo

   If your computer does not have an gpu, check out the branch "no-gpu".

### Processing Scannet Data

1. Download and extract from https://drive.google.com/drive/folders/1V2KIsXIZ-2-5kGDaErTIpRNnBV2zhVjG?usp=sharing:
	a) sample_data.zip (for UprightNet preprocessed ScanNet data)
	b) checkpoints.zip (pretrained weights)
	
  Alternatively you can use 
    d) ScanNet.zip and test_scannet_normal_list.txt from https://drive.google.com/drive/folders/1WdNAESqDYcUPQyXAW6PvlcdQIYlOEXIw instead of a), which contains all scenes.

2. Adapt the paths of sample_data/test_scannet_normal_list.txt to the path of sample_data (e.g. with find&replace)

3. Copy checkpoints folder to root directory of UprightNet repo

4. Adapt DATA_PATH in util/config.py to be the directory where test_scannet_normal_list.txt is located

5. To add the predicted gravity vector to the data folders, run
	python3 test.py --mode ResNet --dataset scannet
	
	Each scene folder should now contain 4 new folders: pose_pred, pose_gt, gravity_pred, gravity_gt

	If your computer does not have an gpu, check out the branch "no-gpu".

---
## Running the pipeline

### Preparing the execution environment
- Go to the `3d-vision` directory

- Install the requirements in your `python 3.9.5` environment
  - `pip install -r requirements.txt`
- Make sure that `poselib` wheel package generated earlier is also installed in this environment

### Running the pipeline (to compare 3-point with 5-point estimator)
- Go to the `3d-vision` folder. There run the command `PYTHONPATH=. python visn/main.py`
- In summary, it does the following
  - loads the dataset (which contains images, corresponding intrinsic matrices, gravity vectors, ground truth relative pose etc. )
  - extracts keypoints and find matches
  - computes relative by running 3-point estimator and using gravity
  - computes relative pose using 5-point estimator
  - compares the runtimes and pose errors for the solutions of both the estimators
  - generates a `.csv` file containing metrics like pose error, runtime, inliers and ransac iterations
    - the output will be saved in a timestamped file in `pipeline_outputs` directory
    - _e.g._ `pipeline_outputs/2022-06-12_201720__run_stats.csv`


> Following are the details of the steps involved

__Data loading__


- `GroupedImagesDataset` from ``visn/data/loader.py`` is the dataset class 
- and `SequentialDataLoader` is the loader used to feed the data batches to the pipeline
- By default dataset from `resources/easy_scenes/scene0025_01` will be loaded.
- To switch dataset folder change `resource_scene` attribute in `visn/config/config.json`
  - snippet from `visn/config/config.json`
    ```json
    "dataset": {
        "object": "GroupedImagesDataset",
        "resource_scene": "easy_scenes/scene0025_01",
        "same_intrinsic_matrix_for_all": true,
        "use_prediction": true
      }
    ```

__Pipeline__

- `BasePipeline` in `visn/process/pipeline.py`, brings all the steps of the pipeline together and provides a shared execution context
- To initialize it loads its config from `visn/config/config.json`
- For each batch of data is applies a sequence of steps on them
  - Note: To make the outputs of previous steps available to the next steos, in our implementation, at each step we process the inputs (which are list of dictionaries) and store the results back in the same dictionary. 
- The following is a snippet from `BasePipeline` which shows the steps through which each batch of data (and results-so-far) will flow
  ```py
  self.steps = [
              AdHocTransforms().process,
              self.angle_manipulator_processor.process,
              self.preprocessor.process,
              self.pose_estimation_processor.process,
              self.benchmarking_processor.process
          ]
  ```
__Ad hoc transforms__

- To transfrom the ScanNet's axis and realtive pose conventions to the convention assumed in our implementation the data is passed through this step of the pipeline. It's defined in `AdHocTransforms` in `visn/process/pipeline.py`

__Angle manipulation__

- (TODO @Yoga)

__Preprocessing__

- This step is defined in `BasePreprocessor` (from `visn/process/components.py`)
- It does the following 
  - normalizes the keypoints using intrisic matrix
  - finds key point matches with the help of `OpenCvKeypointMatcher`
  - computes gravity vectors if required
  - for 3-point based estimation it computes aligned keypoints using gravity information
  - computes ground truth relative pose from the input ground truth absolute pose
- Pipeline then passes the input container (with preprocessor's outputs added in it) to the next step
  
__Pose estimation processing__

- This step is defined in `PoseEstimationProcessor` (from `visn/process/components.py`)
- It prepares RANSAC options, bundle options, camera models for both the 3-point and 5-point estimator
- It feeds the normalized keypoints to  5-point estimator and normalized & aligned keypoints to 3-point estimator, to get the relative pose solutions
- It removes the effect of alignment from the solution of 3-point estimator
- Pipeline then passes the input container (with PoseEstimationProcessor's outputs added in it) to the benchmarking step

__Benchmarking__

- This is defined in `BenchmarkingProcessor` (from `visn/process/components.py`)
- It computes pose estimation runtimes (by averaging `20` trials) 
- It also computes rotation and translation errors for both the estimators

__Summary__

- This step is run only after all the batches have been processed
- It saves all the metrics recorded during the run in a csv file
- It is defined in `BasePipeline.save_summary_csv` (in `visn/process/pipeline.py`)


__Interpreting the results__

(TODO @Timo)

---

## Miscellaneous

### Training UprightNet




