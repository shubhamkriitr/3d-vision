# visn


# Structure of the submitted code

# Building Poselib Python Library

We have added a 3-point estimator to the existing PoseLib library (`TODO:Path`).
The final wheel file will be generated once you executre the following steps. (However we have also attached the final wheel file `TODO:whl file path` so you can quickstart by skipping the following build steps)



# Install the generated wheel file

- Run the following
```sh
```

# Running the pipeline

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



# Interpreting the results


---

# Other steps of the pipeline

# Predicting Gravity


# Processing Scannet Data


