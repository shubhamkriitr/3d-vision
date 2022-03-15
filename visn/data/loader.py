import glob
from typing import Union, List
# TODO: Also see: https://pytorch.org/docs/stable/data.html

class BaseDataLoader(object):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def load(file_path: str, *args, **kwargs):
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
    
    def load(file_path, *args, **kwargs):
        """
        loads a single file and returns a pytorch tensor or a numoy array
        """
        raise NotImplementedError()

if __name__ == "__main__":
    
