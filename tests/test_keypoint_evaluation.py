# Compute the good-bad-keypoint ratio and plot the good keypoints
# This is computed given the true relative pose and making use of the epipolar
# distance, as well as an epsilon threshold which separates good from bad
# keypoints. Currently not reliable due to the used KeypointMatchBenchmarker
# which is faulty.

""" enter parameters """
instance_nr = 0
kp_matching_ratio = 1
epsilon = 50

""" run test """

from visn.data.loader import GroupedImagesDataset
from visn.estimation.keypoint import OpenCvKeypointMatcher, KeypointMatchBenchmarker
import matplotlib.pyplot as plt
import numpy as np

# Setup
dataset = GroupedImagesDataset()
kpm = OpenCvKeypointMatcher({})

# load and process data
data_batch = dataset[instance_nr]
img_1 = data_batch["input_images"][0]
img_2 = data_batch["input_images"][1]
keypoints_1, keypoints_2 = kpm.get_matches(img_1, img_2, ratio=kp_matching_ratio)

# use benchmarker for checking % of good matches (score)
kpt_benchmarker = KeypointMatchBenchmarker(data_batch["input_relative_poses"], data_batch["K"])
inliers, score = kpt_benchmarker.check(keypoints_1, keypoints_2, epsilon=epsilon)
print(f"inlier / outlier Ratio: {score}")

# remove outliers
kp1, kp2 = np.array(keypoints_1), np.array(keypoints_2)
kp1, kp2 = kp1[inliers], kp2[inliers]

# plot only inliers
img = np.concatenate((img_1, img_2), axis=1)
x_offset = img_1.shape[1]
kp2_adj = kp2.copy()
kp2_adj[:, 0] += x_offset
plt.imshow(img)
for ind, (i, j) in enumerate(zip(kp1, kp2_adj)):
    plt.plot([i[0], j[0]], [i[1], j[1]], 'y-')
plt.show()
