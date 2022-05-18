import numpy as np
import shutil
import os
from loguru import logger

DIR_CALIBRATION = "calibration"
DIR_GRAVITY_GT = "gravity_gt"
DIR_GRAVITY_PRED = "gravity_pred"
DIR_IMAGES = "images"
DIR_REL_POSE = "relative_pose"
DIR_ROLL_PITCH_GT = "roll_pitch_gt"
DIR_ROLL_PITCH_PRED = "roll_pitch_pred"

ALL_VISN_DIRS = [
    DIR_CALIBRATION,
    DIR_GRAVITY_GT,
    DIR_GRAVITY_PRED,
    DIR_IMAGES,
    DIR_REL_POSE,
    DIR_ROLL_PITCH_GT,
    DIR_ROLL_PITCH_PRED
]

class DTUtoVisnConverter:
    
    def __init__(self) -> None:
        # Add attrs if required
        pass

    def convert(self, input_dir, target_dir):
        self.init_visn_folder_structure(target_dir)
        image_ids = self.inspect_dtu_data_dir(input_dir)
        for sr_num, image_id in enumerate(image_ids, 1):
            logger.info(f"Processing file #{sr_num} Id: {image_id} ")
            self.process_one_image(sr_num, image_id, input_dir, target_dir)
    
    def inspect_dtu_data_dir(self, dtu_scan_dir):
        """ `images/rect_001_0_r5000.png`
        `cams/00000000_cam.txt`
        """
        image_dir = os.path.join(dtu_scan_dir, "images")
        #>>> cam_info_dir = os.path.join(dtu_scan_dir, "cams")
        image_files = os.listdir(image_dir)
        image_files = [f for f in image_files if f.startswith("rect_")]
        
        image_ids = set([f.split("_")[1] for f in image_files])
        image_ids = sorted(list(image_ids))
        
        logger.info(f"found these image#s: {image_ids}")
        
        return image_ids
    
    
    def process_one_image(self, sr_num, image_id, dtu_scan_dir, target_dir)
        img_file_name = f"rect_{image_id}_6_r5000.png"
        img_path = os.path.join(dtu_scan_dir, "images", img_file_name)
        
        out_sr_num = str(sr_num).zfill(4)
        out_img_path = os.path.join(target_dir, DIR_IMAGES, out_sr_num+".png")
        
        logger.info(f"Copying {img_path} => {out_img_path}")
        shutil.copyfile(img_path, out_img_path)
        
        cam_id = str(int(image_id) - 1).zfill(8)
        cam_file_name = f"{cam_id}_cam.txt"
        cam_file_path = os.path.join(dtu_scan_dir, "cams", cam_file_name)
        
        logger.info(f"Reading cam file from : {cam_file_path}")
        
        Rt, K, size = self.read_cam_file(cam_file_path)
        
    
    def read_cam_file(self, file_loc):
        lines = None
        
        with open(file_loc, "r") as f:
            lines = f.readlines()
        
        idx = 0
        Rt = None
        K = None
        size = None
        item_count = 0
        while idx < len(lines):
            text = lines[idx].strip()
            if item_count == 3:
                break
            if text == "extrinsic":
                Rt_lines = lines[idx+1:idx+1+3] # ignore last row of 4x4 matrxi
                Rt_text = "\n".join(Rt_lines)
                # which contains 0 0 0 1
                idx += 5
                item_count += 1
                continue
            if text == "intrinsic":
                K_lines = lines[idx+1, idx+3]
                K_text = "\n".join(K_lines)
                idx += 4
                item_count += 1
                continue
            
            if item_count == 2: # Rt and K already read
                if text != "":
                    size_text = text
                    item_count +=1
                    continue
            
            
            
            
        
        
    
    def init_visn_folder_structure(self, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for dir_name in ALL_VISN_DIRS:
            os.makedirs(os.path.join(target_dir, dir_name), exist_ok=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", "-i", type=str, default="resources/dtu/scan1")
    ap.add_argument("--output-dir", "-o", type=str, default="resources/dtu_visn_test")
    
    args = ap.parse_args()
    
    converter = DTUtoVisnConverter()
    converter.convert(args.input_dir, args.output_dir)