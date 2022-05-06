import glob
from typing import Union, List
from visn.utils import logger
from visn.constants import RESOURCE_ROOT, DATA_ROOT, IMG_EXT_PNG
from visn import constants
from pathlib import Path
import os
import cv2 as cv
import numpy as np
from typing import Dict, List
from collections import defaultdict
# TODO: Also see: https://pytorch.org/docs/stable/data.html

class BaseDataLoader(object):
    def __init__(self, *args, **kwargs) -> None:
        # specific initialization to be handeled in 
        # derived classes
        pass

    def load(self, file_path: str, *args, **kwargs):
        with open(file_path, "r") as f:
            data = f.read()
        return data

    def load_all(self, location: Union[List[str], str], *args, **kwargs):
        file_paths = self.resolve_paths(location)
        data = []
        for file_path in file_paths:
            current = self.load(file_path)
            data.append(current)
        return data

    def resolve_paths(self, file_paths_or_glob):
        if isinstance(file_paths_or_glob, (list, tuple)):
            return file_paths_or_glob
        elif isinstance(file_paths_or_glob, str):
            return glob.glob(file_paths_or_glob)
    
class ScanNetDataLoader(BaseDataLoader):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def load(self, file_path, *args, **kwargs):
        """
        loads a single file and returns a pytorch tensor or a numoy array
        """
        raise NotImplementedError()


class BaseDataset:
    """
        Given the predefined dataset structure, we load assets of a given scene.
        Dataset Structure:
            resources
            |---calibration
            |    |---image_size.txt
            |    |---K.txt
            |
            |---groups.txt
            |
            |---images
            |    |---0000.png
            |    |---0001.png
            |    |---...
            |
            |----relative_pose
            |    |---0000.txt
            |    |---0001.txt
            |    |---...
            |
            |----roll_pitch_gt
            |    |---0000.txt
            |    |---0001.txt
            |    |---...
            |
            |----roll_pitch_pred
            |    |---0000.txt
            |    |---0001.txt
            |    |---...
    """
    def __init__(self, data_root_dir: str = DATA_ROOT, image_extension: str = IMG_EXT_PNG, **kwargs) -> None:
        self.data_root_dir = data_root_dir
        self.image_extension = image_extension

        # load relevant data
        self.ids = self.get_ids(self.data_root_dir, self.image_extension)
        self.id_digits = len(self.ids[0])
        self.calibration = self.get_calibration(self.data_root_dir)
        self.groups = self.get_groups(self.data_root_dir)

    def get_calibration(self, data_root_dir: str = DATA_ROOT) -> Dict[str, List]:
        # define required paths
        calibration_dir_path = os.path.join(data_root_dir, "calibration")
        image_size_file_path = os.path.join(calibration_dir_path, "image_size.txt")
        k_file_path = os.path.join(calibration_dir_path, "K.txt")

        # load image size
        with open(image_size_file_path, "r") as f:
            content = f.read()
            image_size = [int(x) for x in content.split("\n")]

        # load K
        with open(k_file_path, "r") as f:
            content = f.read()
            k = [[float(y) for y in x.split(" ")] for x in content.split("\n") if x]

        return {"image_size": image_size, "K": k}

    @staticmethod
    def get_groups(data_root_dir: str = DATA_ROOT) -> List[List[str]]:
        """
        File content is assumed to be similar to the example below:
            0000 0001
            0000 0002
            0002 0003
            0004 0005
        """
        groups_file_path = os.path.join(data_root_dir, "groups.txt")
        with open(groups_file_path, "r") as f:
            lines = f.readlines()
        groups = []
        for l in lines:
            l = l.strip().split()
            current_group = [id_.strip() for id_ in l]
            groups.append(current_group)

        return groups

    @staticmethod
    def get_ids(data_root_dir: str = DATA_ROOT, image_extension: str = IMG_EXT_PNG) -> List[str]:
        images_dir_path = os.path.join(data_root_dir, "images")
        files = os.listdir(images_dir_path)
        ids = [name.split(".")[0] for name in files
                     if name.endswith(image_extension)]
        return ids

    def get_id(self, index: int) -> str:
        return str(index).rjust(self.id_digits, '0')

    @staticmethod
    def get_image(id_: str, data_root_dir: str = DATA_ROOT, image_extension: str = IMG_EXT_PNG) -> np.ndarray:
        image_file_path = os.path.join(data_root_dir, "images", f"{id_}{image_extension}")
        img = cv.imread(image_file_path)
        return img

    @staticmethod
    def get_relative_pose(id_: str, data_root_dir: str = DATA_ROOT) -> List[List[float]]:
        relative_pose_file_path = os.path.join(data_root_dir, "relative_pose", f"{id_}.txt")
        with open(relative_pose_file_path, "r") as f:
            content = f.read()
            rel_pose = [[float(y) for y in x.split(" ")] for x in content.split("\n") if x]
        return rel_pose

    @staticmethod
    def get_roll_pitch(id_: str, gth: bool = True, data_root_dir: str = DATA_ROOT) -> List[float]:
        # gth: True => get roll_pitch_gt
        # gth: False => get roll_pitch_pred
        if gth:
            roll_pitch_file_path = os.path.join(data_root_dir, "roll_pitch_gt", f"{id_}.txt")
        else:
            roll_pitch_file_path = os.path.join(data_root_dir, "roll_pitch_pred", f"{id_}.txt")
        with open(roll_pitch_file_path, "r") as f:
            content = f.read()
            rel_pose = [float(x) for x in content.split(" ")]
        return rel_pose

    @staticmethod
    def get_gravity(id_: str, gth: bool = True, data_root_dir: str = DATA_ROOT) -> List[float]:
        # gth: True => get gravity_gt
        # gth: False => get gravity_pred
        if gth:
            gravity_file_path = os.path.join(data_root_dir, "gravity_gt", f"{id_}.txt")
        else:
            gravity_file_path = os.path.join(data_root_dir, "gravity_pred", f"{id_}.txt")
        with open(gravity_file_path, "r") as f:
            content = f.read()
            rel_pose = [float(x) for x in content.split(" ")]
        return rel_pose

    def __getitem__(self, id_: str):
        # load and return image, relative_pose, roll_pitch_gt and roll_pitch_pred
        out = {"img": self.get_image(id_, self.data_root_dir, self.image_extension),
               "rel_pose": self.get_relative_pose(id_, self.data_root_dir),
               "rp_gt": self.get_roll_pitch(id_, gth=True),
               "rp_pred": self.get_roll_pitch(id_, gth=False),
               "gr_gt": self.get_gravity(id_, gth=True),
               "gr_pred": self.get_gravity(id_, gth=False)}
        return out
    
    def __len__(self):
        return len(self.ids)


class GroupedImagesDataset(BaseDataset):
    """Image pair data set. Loads individual images and K matrices as array
    from specified data rootdir, applies
    random transformations to create synthetic pairs to be used later in the
    pipeline.
    Assumes images are named `<id>.<extension>` and corresponding intrinsic
    matrix is `K_<id>.txt` (otherwise `K.txt` will be used).
    Image grouping is read from `GROUPS.txt`
    """

    def __init__(self, data_root_dir: str = DATA_ROOT, image_extension: str = IMG_EXT_PNG, use_prediction: bool = True,
                 **kwargs) -> None:
        super().__init__(data_root_dir, image_extension)
        self.use_prediction = use_prediction
        self.metadata = ""  # to be used later/ for specifying modalities and versions etc.

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int):
        # get relevant data for group
        group = self.groups[index]
        img_, rel_pos_, rp_gt_, rp_pred_ = [], [], [], []
        gr_gt_, gr_pred_, gr_, k_ = [], [], [], []
        for id_ in group:
            img, rel_pose, rp_gt, rp_pred, gr_gt, gr_pred = super().__getitem__(id_).values()
            img_.append(img)
            rel_pos_.append(rel_pose)
            rp_gt_.append(rp_gt)
            rp_pred_.append(rp_pred)
            gr_gt_.append(gr_gt)
            gr_pred_.append(gr_pred)
            k_.append(self.calibration["K"])
        gr_ = gr_pred_ if self.use_prediction else gr_gt_

        # structure output
        out = {"input_images": img_,
               "input_relatives_poses": rel_pos_,
               "input_roll_pitch_gt": rp_gt_,
               "input_roll_pitch_pred": rp_pred_,
               "input_gravity_gt": gr_gt_,
               "input_gravity_pred": gr_pred_,
               "input_gravity": gr_,
               "K": k_}
        return out

        
class SequentialDataLoader(object): # TODO: may use torch's loader instead
    def __init__(self, dataset, batch_size) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self._num_samples_yieled = 0
        
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._num_samples_yieled >= len(self.dataset):
            self._num_samples_yieled = 0
            raise StopIteration
        start = self._num_samples_yieled
        self._num_samples_yieled \
            = min(self._num_samples_yieled + self.batch_size,
                  len(self.dataset))
        
        batch_data = []
        for idx in range(start, self._num_samples_yieled):
            batch_data.append(self.dataset[idx])
        return batch_data
            
        

if __name__ == "__main__":
    # loader = BaseDataLoader()
    dataset = GroupedImagesDataset()
    loader = SequentialDataLoader(dataset=dataset, batch_size=1)
    for idx, data in enumerate(loader):
        d = data
        print(data)
    

