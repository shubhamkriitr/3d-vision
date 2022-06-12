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
- Extract `eigen-3.4.0.zip` to the current folder (Note: `eigen-3.4.0.zip` was downloaded from https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip )
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

### Processing Scannet Data


---
## Running the pipeline

### Preparing the execution environment
- Go to the `3d-vision` (TODO-name it correctly) directory
- Execute the following in sequence (enter yes when prompted):
```
conda create -n 3dvis-py3-9-5 python==3.9.5
conda activate 3dvis-py3-9-5
pip install -r requirements.txt
```
- Now the environment should be ready
- Make sure to check that the environment is activated before running the code
stt
#### Install the generated wheel file

- Run the following
```sh
```

### Summary of pipeline steps
- Go to the `threedvis` folder. There run the command `PYTHONPATH=. python visn/main.py`
  - This will run the whole pipeline which has the following steps
    - It will load the data
    - After data loading the keypoints for the image pair is extracted and correspondences are found
    - The for the 3 point estimator - these keypoints are transformed so the gravity vector in these two images moves to the y-axis (which, by the 3 point estimator, is assumed to be the common axis of rotation.)
    - Then the keypoints are passed to the corrsponding estimators which return the estimated pose
    - The result of the 3 point estimator is transformed again so as to remove the effect of alignment done earlier
      - During the execution , their runtimes are also recorde for comparison
    - Pose error is computed w.r.t. the ground truth pose for both sets of the estimations
    - These errors , runtimes, and RANSAC parameters are logged into a csv file

### Interpreting the results


---

## Miscellaneous

### Training UprightNet




