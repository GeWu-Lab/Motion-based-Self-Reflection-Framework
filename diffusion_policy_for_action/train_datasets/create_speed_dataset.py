import json
import h5py
import os
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
import shutil

data_name = [
    "coffee_d0"
    # ,"coffee_d1",
    # "stack_d0","stack_d1",
    # "stack_three_d0","stack_three_d1",
    # "threading_d0",
    # "three_piece_assembly_d0","three_piece_assembly_d1",
]
origin_data_dir = "origin_datasets"
aligned_data_dir = "adjust_llava_motion"

with open(os.path.join(aligned_data_dir, "language_idx.json"), "r") as f:
    language_idx = json.load(f)["language_idx"]

source_path = os.path.join(aligned_data_dir, "language_idx.json")
destination_path = os.path.join("speed_adjust_llava_motion", "language_idx.json")

# 复制文件
shutil.copyfile(source_path, destination_path)

def process_name(name):
    origin_data = h5py.File(f"{origin_data_dir}/{name}.hdf5", "r")["data"]
    action_data = h5py.File(f"{aligned_data_dir}/{name}_adjust_llava_motion.hdf5", "r")["data"]
    save_data = h5py.File(f"speed_adjust_llava_motion/{name}_speed_adjust_llava_motion.hdf5", "w")
    data_group = save_data.create_group("data")
    
    start_time = time.time()
    for demo_id in range(1, 501):
        demo_group = data_group.create_group(f"demo_{demo_id}")
        agentview_images = origin_data[f"demo_{demo_id}"]["obs"]["agentview_image"][:]
        ego_images = origin_data[f"demo_{demo_id}"]["obs"]["robot0_eye_in_hand_image"][:]
        ee_pos = origin_data[f"demo_{demo_id}"]["obs"]["robot0_eef_pos"][:]
        ee_quat = origin_data[f"demo_{demo_id}"]["obs"]["robot0_eef_quat"][:]
        gripper = origin_data[f"demo_{demo_id}"]["obs"]["robot0_gripper_qpos"][:]
        
        language_feature = action_data[f"demo_{demo_id}"]["language_feature"][:]
        language = action_data[f"demo_{demo_id}"]["language"][:]
        action_chunking = action_data[f"demo_{demo_id}"]["action_chunking"][:]

        data_len = len(language_feature)
        
        for timestep in range(data_len):
            timestep_group = demo_group.create_group(str(timestep))
            timestep_group.create_dataset("agentview_image", data=agentview_images[timestep])
            timestep_group.create_dataset("ego_image", data=ego_images[timestep])
            timestep_group.create_dataset("ee_pos", data=ee_pos[timestep])
            timestep_group.create_dataset("ee_quat", data=ee_quat[timestep])
            timestep_group.create_dataset("gripper", data=gripper[timestep])
            timestep_group.create_dataset("language_feature", data=language_feature[timestep])
            timestep_group.create_dataset("language_idx", data = language_idx[language[timestep].decode('utf-8')])
            timestep_group.create_dataset("language", data=language[timestep])
            timestep_group.create_dataset("action_chunking", data=action_chunking[timestep])
    
        print(f"{name} processing time: {time.time() - start_time:.2f} seconds", "the idx is:", demo_id)
    save_data.close()
    origin_data.file.close()
    action_data.file.close()



# Use ProcessPoolExecutor to process each name in parallel
with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = [executor.submit(process_name, name) for name in data_name]

    # Ensure all processes have completed
    for future in futures:
        future.result()
