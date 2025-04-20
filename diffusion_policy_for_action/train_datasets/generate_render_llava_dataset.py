import torch
import numpy as np
import mimicgen_envs
import robosuite as suite
import cv2
import os
import imageio
from robosuite.controllers import load_controller_config
import json

import h5py
from tqdm import tqdm

TASK_MAPPING = {
    
    
    "coffee_d0": ["Coffee_D0", "make coffee"],
    # "coffee_d1": ["Coffee_D1", "make coffee"],
    # "stack_d0": ["Stack_D0", "stack the red block on top of the green block"],
    # "stack_d1": ["Stack_D1", "stack the red block on top of the green block"],
    # "stack_three_d0": ["StackThree_D0", "stack the blocks in the order of blue, red, and green from top to bottom"],
    # "stack_three_d1": ["StackThree_D1", "stack the blocks in the order of blue, red, and green from top to bottom"],
    # "threading_d0": ["Threading_D0", "insert the needle into the needle hole"],
    # "three_piece_assembly_d0":["ThreePieceAssembly_D0","stack the three pieces"],
    # "three_piece_assembly_d1":["ThreePieceAssembly_D1","stack the three pieces"],
}


if __name__ == "__main__":
    
    for key in TASK_MAPPING:

        print("the key is:", key)
        options = {}

        # Choose environment
        options["env_name"] = TASK_MAPPING[key][0]

        # Choose robot
        options["robots"] = "Panda"

        # Load the desired controller
        options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")

        env = suite.make(
            **options,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=False,
            use_camera_obs=True,
            control_freq=20,
            camera_heights=84,
            camera_widths=84,
            reward_shaping= False,
            camera_names= ["agentview","robot0_eye_in_hand"]
        )
        obs_action_datapath = "./origin_datasets/{}.hdf5".format(key)

        data = h5py.File(obs_action_datapath,"r")['data']
        
        json_data = []
        idx = 0
        
        dir_path = os.path.join("./generate_from_state",key)
        os.makedirs(dir_path)
        
        
        for demo_id in tqdm(range(501)):

            env.reset()
            manipulation_data = data[f'demo_{demo_id}']
            manipulation_length = len(manipulation_data["states"])
            for state_idx in range(manipulation_length):
                idx += 1
                demo_state = manipulation_data["states"][state_idx]
                env.sim.set_state_from_flattened(demo_state)
                env.sim.forward()
                
                im = env.sim.render(height=336, width=336, camera_name="agentview")
                rgb = im[::-1]
                # obs, reward, done, _ = env.step(action)


                imageio.imsave(f"{dir_path}/{key}_{idx}.jpg", rgb)
                # print("the robot langage is:", language)
                single_data = {
                    "id": idx,
                    "image": f"{dir_path}/{key}_{idx}.jpg",
                    "conversations":[
                        {
                            "from": "human",
                            "value": f"<image>\nSuppose you are the robot to {TASK_MAPPING[key][1]}, what would you do next?"
                        },
                        {
                            "from": "gpt",
                            "value": TASK_MAPPING[key][1]
                        }
                    ]
                }
                json_data.append(single_data)
                print("the idx is:", idx)
                
                
        with open(f"./generate_from_state/{key}_data.json", "w") as f:
            json.dump(json_data, f, indent=4)