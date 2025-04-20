import numpy as np
import json
import h5py
import cv2
import os

TASK_MAPPING = {
    "coffee_d0": ["Coffee_D0", "make coffee"],
    # "coffee_d1": ["Coffee_D1", "make coffee"],
    # "stack_d0": ["Stack_D0", "stack the red block on top of the green block"],
    # "stack_d1": ["Stack_D1", "stack the red block on top of the green block"],
    # "stack_three_d0": ["StackThree_D0", "stack the blocks in the order of blue, red, and green from top to bottom"],
    # "stack_three_d1": ["StackThree_D1", "stack the blocks in the order of blue, red, and green from top to bottom"],
    # "threading_d0": ["Threading_D0", "insert the needle into the needle hole"],
    # "square_d0": ["Square_D0", "slide the square block onto the wooden stick"],
    # "three_piece_assembly_d0":["ThreePieceAssembly_D0","stack the three pieces"],
    # "three_piece_assembly_d1":["ThreePieceAssembly_D1","stack the three pieces"],
    # "mug_cleanup_d0":["MugCleanup_D0","put the mug into the drawer"],
    # "mug_cleanup_d1":["MugCleanup_D1","put the mug into the drawer"],
}


json_data = []
if __name__ == "__main__":

    total_idx = 0
    for name in TASK_MAPPING:
        input_name = name.lower()
        data_path = os.path.join("./adjust_llava_motion", f"{input_name}_adjust_llava_motion.hdf5")
        with h5py.File(data_path, 'r') as f:
            idx = 0
            task = TASK_MAPPING[input_name][1]
            # image_name = input_name
            # image_name = TASK_MAPPING[input_name][0]
            image_name = name
            image_path = name

                
            for num in range(500):

                robot_language = f['data'][f'demo_{num}']['language']
                print("the language length is:", len(robot_language))
                for i in range(len(robot_language)):
                    idx += 1
                    # input_image_path = f"render_llava_datasets/{image_path}/{image_name}_{idx}.jpg"

                    total_idx += 1
                    single_data = {
                        "id": total_idx,
                        "image": f"{image_path}/{image_name}_{idx}.jpg",
                        "conversations":[
                            {
                                "from": "human",
                                "value": f"<image>\nSuppose you are the robot to {task}, what would you do next?"
                            },
                            {
                                "from": "gpt",
                                "value": robot_language[i].decode('utf-8')
                            }
                        ]
                    }
                    json_data.append(single_data)
                print("the idx is:", total_idx)


    with open("./adjust_llava_motion/llava_motion.json", "w") as f:
        json.dump(json_data, f, indent=4)