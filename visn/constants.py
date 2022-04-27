from pathlib import Path
from datetime import datetime

# to be used as default paths if paths not resolved using the config
# provide during running the pipeline

MAIN_MODULE_ROOT = Path(__file__).parent
PROJECTPATH = MAIN_MODULE_ROOT.parent
RESOURCE_ROOT = Path(PROJECTPATH)/"resources"
OUTPUT_ROOT = Path(PROJECTPATH)/"outputs"
TEMP_ROOT = Path(PROJECTPATH)/"__temp"

#Image Extensions
IMG_EXT_PNG = ".png"
IMG_EXT_JPEG = ".jpeg"
