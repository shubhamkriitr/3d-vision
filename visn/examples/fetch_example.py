import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

def ReadFeatureMatches(image_pair, data_folder):
  im1 = image_pair[0]
  im2 = image_pair[1]
  assert(im1 < im2)
  matchfile_path = os.path.join(data_folder, 'matches', im1 + '-' + im2 + '.txt')
  pair_matches = np.loadtxt(matchfile_path, dtype=int)
  return pair_matches

def ReadKMatrix(data_folder):
  path = os.path.join(data_folder, 'images', 'K.txt')
  K = np.loadtxt(path)
  return K

def GetPairMatches(im1, im2, matches):
  if im1 < im2:
    return matches[(im1, im2)]
  else:
    return np.flip(matches[(im2, im1)], 1)

# Simple image class to hold some information for the SfM task.
class Image:
  # Constructor that reads the keypoints and image file from the data directory
  def __init__(self, data_folder, name):
      image_path = os.path.join(data_folder, 'images', name)
      keypoints_path = os.path.join(data_folder,'keypoints', name + '.txt')

      self.name = name
      self.image = plt.imread(image_path)
      self.kps = np.loadtxt(keypoints_path)
      self.p3D_idxs = {}

  # Set the image pose.
  # This is assumed to be the transformation from global space to image space
  def SetPose(self, R, t):
    self.R = R
    self.t = t

  # Get the image pose
  def Pose(self):
    return self.R, self.t

  # Add a new 2D-3D correspondence to the image
  # The function expects two equal length lists of indices, the first one with the
  # keypoint index in the image, the second one with the 3D point index in the reconstruction.
  def Add3DCorrs(self, kp_idxs, p3D_idxs):
    for corr in zip(kp_idxs, p3D_idxs):
      self.p3D_idxs[corr[0]] = corr[1]

  # Get the 3D point associated with the given keypoint.
  # Will return -1 if no correspondence is set for this keypoint.
  def GetPoint3DIdx(self, kp_idx):
    if kp_idx in self.p3D_idxs:
      return self.p3D_idxs[kp_idx]
    else:
      return -1

  # Get the number of 3D points that are observed by this image
  def NumObserved(self):
    return len(self.p3D_idxs)


def EstimateEssentialMatrix(K, im1, im2, matches):
  # TODO
  # Normalize coordinates (to points on the normalized image plane)

  # These are the keypoints on the normalized image plane (not to be confused with the normalization in the calibration exercise)
  K_inv = np.linalg.inv(K)
  normalized_kps1 = np.append( im1.kps, np.ones((im1.kps.shape[0],1)), 1) @ (K_inv.transpose())
  normalized_kps2 = np.append( im2.kps, np.ones((im2.kps.shape[0],1)), 1) @ (K_inv.transpose())

  # # TODO
  # # Assemble constraint matrix
  # constraint_matrix = np.zeros((matches.shape[0], 9))
  
  # for i in range(matches.shape[0]):
  #   # TODO
  #   # Add the constraints
  #   val = normalized_kps1[matches[i,0]][:,None] @ normalized_kps2[matches[i,1]][None,:]
  #   constraint_matrix[i] = val.flatten()
  kps1 = normalized_kps1[matches[:,0]]
  kps2 = normalized_kps2[matches[:,1]]
  constraint_matrix = np.matmul(kps1[:,:,None],kps2[:,None,:])
  constraint_matrix = constraint_matrix.reshape(-1,9)
  
  # Solve for the nullspace of the constraint matrix
  _, _, vh = np.linalg.svd(constraint_matrix)
  vectorized_E_hat = vh[-1,:]

  # TODO
  # Reshape the vectorized matrix to it's proper shape again
  E_hat = vectorized_E_hat.reshape(3,3)

  # TODO
  # We need to fulfill the internal constraints of E
  # The first two singular values need to be equal, the third one zero.
  # Since E is up to scale, we can choose the two equal singluar values arbitrarily
  u, s, vh = np.linalg.svd(E_hat)
  print(s)
  s[0] = 1
  s[1] = 1
  s[2] = 0
  E = u @ (s[:,None]*vh)

  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.
  for i in range(matches.shape[0]):
    kp1 = normalized_kps1[matches[i,0],:]
    kp2 = normalized_kps2[matches[i,1],:]
    # print(i, abs(kp1.transpose() @ E @ kp2), abs(kp1.transpose() @ E.transpose() @ kp2))
    assert(abs(kp1.transpose() @ E @ kp2) < 0.01)

  return E


def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols



def TriangulateCheck(K,R,t,im1,im2, matches):
  P1 = K @ np.append(R, np.expand_dims(t, 1), 1)
  P2 = np.append(K, np.zeros((3, 1)), 1) 

  # # Iterative process
  # A = np.zeros((4,4))
  # points3D_2 = np.zeros((matches.shape[0],4))
  # for i in range(matches.shape[0]):
  #   kp1 = im1.kps[matches[i, 0], :]
  #   kp2 = im2.kps[matches[i, 1], :]

  #   # H & Z Sec. 12.2
  #   A[0] = kp1[0] * P1[2] - P1[0]
  #   A[1] = kp1[1] * P1[2] - P1[1]
  #   A[2] = kp2[0] * P2[2] - P2[0]
  #   A[3] = kp2[1] * P2[2] - P2[1]

  #   _, _, vh = np.linalg.svd(A)
  #   homogeneous_point = vh[-1]
  #   points3D_2[i] = homogeneous_point / homogeneous_point[-1]
  
  # Single line process
  A = np.zeros((matches.shape[0],4,4))
  kp1 = im1.kps[matches[:, 0], :]
  kp2 = im2.kps[matches[:, 1], :]

  # H & Z Sec. 12.2
  A[:,0] = kp1[:,0][:,None] * P1[2][None,:] - P1[0][None,:]
  A[:,1] = kp1[:,1][:,None] * P1[2][None,:] - P1[1][None,:]
  A[:,2] = kp2[:,0][:,None] * P2[2][None,:] - P2[0][None,:]
  A[:,3] = kp2[:,1][:,None] * P2[2][None,:] - P2[1][None,:]

  _, _, vh = np.linalg.svd(A)
  homogeneous_point = vh[:,-1]
  points3D_2 = homogeneous_point / homogeneous_point[:,-1][:,None]

  points3D_1 = points3D_2 @ np.append(R, np.expand_dims(t, 1), 1).transpose()
  
  return np.sum( (points3D_1[:,2]>0)&(points3D_2[:,2]>0) )
  

def get_matched_kps(selected_images = [6,5], data_folder = './visn/examples'):

  flipped = (selected_images[0] > selected_images[1])
  if flipped:
    selected_images = np.flip(selected_images)

  image_names = [
    '0000.png',
    '0001.png',
    '0002.png',
    '0003.png',
    '0004.png',
    '0005.png',
    '0006.png',
    '0007.png',
    '0008.png',
    '0009.png']

  im1_name = image_names[selected_images[0]]
  im2_name = image_names[selected_images[1]]

  # Read images
  im1 = Image(data_folder, im1_name)
  im2 = Image(data_folder, im2_name)

  # Read the matches
  image_pair = [im1_name, im2_name]
  matches = ReadFeatureMatches(image_pair, data_folder)

  K = ReadKMatrix(data_folder)

  K_inv = np.linalg.inv(K)
  normalized_kps1 = np.append( im1.kps, np.ones((im1.kps.shape[0],1)), 1) @ (K_inv.transpose())
  normalized_kps2 = np.append( im2.kps, np.ones((im2.kps.shape[0],1)), 1) @ (K_inv.transpose())
  
  matched_kps1 = normalized_kps1[matches[:,0]]
  matched_kps2 = normalized_kps2[matches[:,1]]

  E = EstimateEssentialMatrix(K, im1, im2, matches)
  
  possible_relative_poses = DecomposeEssentialMatrix(E)

  vals = np.zeros(len(possible_relative_poses))
  for i,(R,t) in enumerate(possible_relative_poses):
    vals[i] = TriangulateCheck(K,R,t,im1, im2, matches)
  
  R,t = possible_relative_poses[np.argmax(vals)]

  if flipped:
    return (matched_kps2, matched_kps1, R, t, E)
  else:
    return (matched_kps1, matched_kps2, R, t, E)

def get_image(selected_images = 6, data_folder = './visn/examples'):
  image_names = [
    '0000.png',
    '0001.png',
    '0002.png',
    '0003.png',
    '0004.png',
    '0005.png',
    '0006.png',
    '0007.png',
    '0008.png',
    '0009.png']

  return Image(data_folder, image_names[selected_images])

if __name__ == '__main__':
  a,b,c,d,e = get_matched_kps()

  print(a.shape, b.shape, c.shape, d.shape, e.shape)