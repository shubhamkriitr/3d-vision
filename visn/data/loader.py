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
        self.calibration = self.get_calibration(self.data_root_dir)
        self.groups = self.get_groups(self.data_root_dir)
        self.ids = self.get_ids(self.data_root_dir, self.image_extension)
        self.id_digits = len(self.ids[0])

    @staticmethod
    def get_calibration(data_root_dir: str = DATA_ROOT) -> Dict[str, List]:
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

        return {"image_size": image_size, "k": k}

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

    def __getitem__(self, index):
        # load and return image, relative_pose, roll_pitch_gt and roll_pitch_pred
        id_ = self.get_id(index)
        out = {"img": self.get_image(id_, self.data_root_dir, self.image_extension),
               "rel_pose": self.get_relative_pose(id_, self.data_root_dir),
               "rp_gt": self.get_roll_pitch(id_, gth=True),
               "rp_pred": self.get_roll_pitch(id_, gth=False)}
        return out
    
    def __len__(self):
        return len(self.ids)


class GroupedImagesDataset:
    """Image pair data set. Loads individual images and K matrices as array
    from specified data rootdir, applies
    random transformations to create synthetic pairs to be used later in the
    pipeline.
    Assumes images are named `<id>.<extension>` and corresponding intrinsic
    matrix is `K_<id>.txt` (otherwise `K.txt` will be used).
    Image grouping is read from `GROUPS.txt`
    """
    def __init__(self, config = None, **kwargs) -> None:
        self.image_ids = None
        self.image_extension = None
        self.metadata = "" # to be used later/ for specifying modalities and
        # versions etc.
        self._init_from_config(config)
        self._inspect_data_rootdir()
    
    def _init_from_config(self, config):
        if config is None:
            self.config = {
                "data_rootdir": Path(RESOURCE_ROOT)/"1"/"images",
                "image_extension": constants.IMG_EXT_PNG
            }
        else:
            self.config = config

        self.data_rootdir = self.config["data_rootdir"]
        self.image_extension = self.config["image_extension"]
        
    def _inspect_data_rootdir(self):
        files = os.listdir(self.data_rootdir)
        self.image_ids = [name.split(".")[0] for name in files 
                       if name.endswith(self.image_extension)]
        groups_file_path = str(self.data_rootdir/"GROUPS.txt")
        self.image_groups = self._read_image_groups(groups_file_path)
        
        logger.debug(f"Files: {files}")
        
    def _read_image_groups(self, file_path):
        """File content is assumed to be similar to the example below:
        0000 0001
        0000 0002
        0002 0003
        0004 0005
        """
        lines = []
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        groups = []
        for l in lines:
            l = l.strip().split()
            current_group = [id_.strip() for id_ in l]
            groups.append(current_group)
        
        return groups
        
    def __len__(self) -> int:
        return len(self.image_groups)

    def __getitem__(self, index: int):
        group = self.image_groups[index]
        input_images = []
        intrinsic_matrices = []
        for id_ in group:
            img, k = self.load_data_by_id(id_)
            input_images.append(img)
            intrinsic_matrices.append(k)
        return {"input_images": input_images, "K": intrinsic_matrices}


    def load_data_by_id(self, id_):
        image_path = str(self.data_rootdir/f"{id_}{self.image_extension}")
        k_path = str(self.data_rootdir/f"K_{id}.txt")
        if not os.path.isfile(k_path):
            k_path = self.data_rootdir/f"K.txt"
        k_intrinsic = self.load_intrinsic_matrix(k_path)
        img = cv.imread(image_path)
        # return image as is (i.e. color)
        # >>> img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        return img, k_intrinsic
    
    def load_intrinsic_matrix(self, file_path):
        return np.loadtxt(file_path)

        
        
        
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
    

