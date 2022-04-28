import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from mpl_toolkits.mplot3d import Axes3D

# import open3d as o3d

class BaseVisualizer:
    
    def get_new_subplot(self, plot_type="3d"):
        fig = plt.figure()
        if plot_type == "3d":
            ax = fig.add_subplot( projection="3d")
        else:
            ax = fig.add_subplot(111)
        
        return ax
        
        
    def plot_vectors(self, vectors, origins=None, length=1, ax=None):

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
        ax.quiver(*origins.T, *vectors.T, length=length, normalize=True)

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
        
    

if __name__ == "__main__":
    x = np.array([[1, 1, 1], [1, 2, 2]])
    origins = np.zeros_like(x)
    vis = BaseVisualizer()
    vis.plot_vectors(x, length=1.0)
    plt.show()
    
        
        
                
        
        
        
        
        