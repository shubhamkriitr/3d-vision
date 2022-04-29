import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from mpl_toolkits.mplot3d import Axes3D
import math
# import open3d as o3d
DEFAULT_SUBPLOT_CODE = 111

class BaseVisualizer:
    
    def get_new_subplot(self, plot_type="3d"):
        fig = plt.figure()
        if plot_type == "3d":
            ax = fig.add_subplot( projection="3d")
        else:
            ax = fig.add_subplot(111)
        
        return ax
        
        
    def plot_vectors(self, vectors, origins=None, length=1, ax=None, **kwargs):

        if ax is None:
            ax = self.get_new_subplot()
        
        if origins is None:
            origins = np.zeros_like(vectors)

        ranges_1 = self._compute_limits(origins + vectors)
        ranges_2 = self._compute_limits(origins)
        ranges = self._merge_limit_ranges(ranges_1, ranges_2)
        
        ax.set_xlim(ranges[0])
        ax.set_ylim(ranges[1])
        ax.set_zlim(ranges[2])
        self.mark_axes_3d(ax)
        ax.quiver(*origins.T, *vectors.T, length=length, normalize=True,
                  **kwargs)

    def _merge_limit_ranges(self, ranges_1, ranges_2):
        ranges = []
        for idx in range(len(ranges_1)):
            r1 = ranges_1[idx]
            r2 = ranges_2[idx]
            r = [min(r1[0], r2[0]), max([r1[1], r2[1]])]
            if len(r1) > 2:
                r.extend(min(r1[2], r2[2]))
            ranges.append(r)
        return ranges
        
        
        
    def _compute_limits(self, vectors, margin=0.1, step=None):
        ranges = []
        mins = np.min(vectors, axis=0)*(1 - margin)
        maxes = np.max(vectors, axis=0)*(1 + margin)
        for dim in range(len(mins)):
            r = [mins[dim], maxes[dim]]
            if step is not None:
                r.append(step)
            ranges.append(tuple(r))
        
        return ranges

    def mark_axes_3d(self, ax):
        ax.set_zlabel('${z}$', fontsize=10, rotation = 0)
        ax.set_ylabel('${y}$', fontsize=10, rotation = 0)
        ax.set_xlabel('${x}$', fontsize=10, rotation = 0)

    def plot_3d_points(self, points, ax=None):
      if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(DEFAULT_SUBPLOT_CODE, projection='3d')
      ax.plot(xs=points[:,0], ys=points[:,1], zs=points[:,2],
              color='g', marker='.', linestyle='None')
    
      plt.show(block=False)

      return ax
  

    def plot_camera(self, R, c, ax=None, f=1, hx=1, hy=4, scale=1.0, color='b'):
      if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(DEFAULT_SUBPLOT_CODE, projection='3d')

      rotation_inv = R.transpose()
      t = -rotation_inv @ c

      # focal length f
      # half width  hx
      # half height hy

      camera_points = np.array([
        [ hx,  hy, f],
        [-hx,  hy, f],
        [-hx, -hy, f],
        [ 0 ,  0 , 0],
        [-hx,  hy, f],
        [ hx, -hy, f],
        [ 0 ,  0 , 0],
        [ hx,  hy, f],
        [ hx, -hy, f],
        [-hx, -hy, f],
        [ hx,  hy, f],
      ]).transpose()

      t = np.reshape(t, (3, 1))

      world_points = (rotation_inv @ (scale * camera_points) + t).T

      ax.plot(xs=world_points[:,0], ys=world_points[:,1], zs=world_points[:,2],
              color=color)

      plt.show(block=False)

      return ax


    def plot_2d_points(self, points, image_shape, ax=None):
      if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(DEFAULT_SUBPLOT_CODE)

      corners = np.array([
        [0, 0],
        [image_shape[0], 0],
        [image_shape[0], image_shape[1]],
        [0, image_shape[1]],
      ])
      ax.plot(corners[:,0], corners[:,1])
      ax.plot(points[:,0], image_shape[1] - points[:,1], 'r.')

      plt.show(block=False)


    def plot_projected_points(self, points_3d, points_2d, K, R, t,
                              image_shape, ax=None,
                              color="r"):
      if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(DEFAULT_SUBPLOT_CODE)

      # project to image coordinates
      projected_2d = K @ ((R @ points_3d.transpose()) + t)
      projected_2d = projected_2d[0:2, :] / projected_2d[[-1],:]

      ax.plot(projected_2d[0,:], image_shape[1] - projected_2d[1,:], 'r.')
      num_points = points_2d.shape[0]
      for i in range(num_points):
        ax.plot([projected_2d[0,i], points_2d[i,0]],
                [image_shape[1] - projected_2d[1,i],
                image_shape[1] - points_2d[i,1]],
                color=color)

      plt.show(block=False)

    def plot_images(self, images):
      num_images = len(images)

      grid_height = math.floor(math.sqrt(num_images))
      grid_width = math.ceil(num_images / grid_height)

      fig = plt.figure()

      for idx in enumerate(images):
        ax = fig.add_subplot(grid_height, grid_width, idx+1)
        ax.imshow(images[idx])

      plt.show(block=False)


    def plot_image_with_keypoints(self, image, keypoints):
      fig = plt.figure()
      ax = fig.add_subplot(DEFAULT_SUBPLOT_CODE)
      ax.imshow(image)
      ax.plot(keypoints[:,0], keypoints[:,1], 'b.')
      plt.show(block=False)


    def plot_image_pair_keypoint_matches(self, image_1, image_2,
                                         keypoints_1, keypoints_2, ax=None,
                                         style="b-"):


      offset = image_1.shape[1]
      image_pair = np.concatenate([image_1, image_2], axis=1)
      if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(DEFAULT_SUBPLOT_CODE)
      ax.imshow(image_pair)
      #plot keypoints
      ax.plot(image_1.kps[:,0], image_1.kps[:,1], 'r.')
      ax.plot(image_2.kps[:,0] + offset, image_2.kps[:,1], 'r.')
    
      # connect keypoints
      for i in range(keypoints_1.shape):
        ax.plot([keypoints_1[i][0], keypoints_2[i][0] + offset],
                [keypoints_1[i][1], keypoints_2[i][1]], style,
                linewidth=0.8)
      plt.show()



if __name__ == "__main__":
    x = np.array([[1, 1, 1], [1, 2, 2]])
    origins = np.zeros_like(x)
    vis = BaseVisualizer()
    vis.plot_vectors(x, length=1.0)
    plt.show()
    
        
        
                
        
        
        
        
        