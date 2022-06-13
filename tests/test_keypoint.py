# Plot keypoints of instance-pair for sanity-check
# Enter params to get desired instance-pair and configurations

""" enter parameters """

intance_nr = 1
kp_matching_ratio = 0.75
every_x_th_kp_match = 5  # plot only every x_th keypoint (for overview)
point_size = 1
point_color = 'y'

""" run test """

from visn.data.loader import GroupedImagesDataset
from visn.estimation.keypoint import OpenCvKeypointMatcher
import matplotlib.pyplot as plt
import numpy as np

dataset = GroupedImagesDataset()
kpm = OpenCvKeypointMatcher({})
data_batch = dataset[intance_nr]

# load and process data
img_1 = data_batch["input_images"][0]
img_2 = data_batch["input_images"][1]
keypoints_1, keypoints_2 = kpm.get_matches(img_1, img_2, ratio=kp_matching_ratio)
kp1 = np.array(keypoints_1)
kp2 = np.array(keypoints_2)

# plot
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

# plot axis 1
img = np.concatenate((img_1, img_2), axis=1)
x_offset = img_1.shape[1]
kp2_adj = kp2.copy()
kp2_adj[:, 0] += x_offset
ax1.imshow(img)
ax1.scatter(kp1[:, 0], kp1[:, 1], s=point_size, c=point_color)
ax1.scatter(kp2_adj[:, 0], kp2_adj[:, 1], s=point_size, c=point_color)

# plot axis 2
ax2.imshow(img)
for ind, (i, j) in enumerate(zip(kp1, kp2_adj)):
    if ind % every_x_th_kp_match == 0:  # sample less for overview
        ax2.plot([i[0], j[0]], [i[1], j[1]], point_color + '-')
plt.show()
