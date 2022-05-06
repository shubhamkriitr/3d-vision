from visn.constants import MAIN_MODULE_ROOT
import json
import os

CONFIG_EXT_JSON = ".json"


def read_config(config_name: str = "config"):
    config_name += CONFIG_EXT_JSON
    config_path = os.path.join(MAIN_MODULE_ROOT, "config", config_name)
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
