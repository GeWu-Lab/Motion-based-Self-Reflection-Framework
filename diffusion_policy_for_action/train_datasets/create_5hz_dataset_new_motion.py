import h5py
import cv2
import numpy as np
import clip
import torch


def fill_action(action, action_length):
    padding_action = np.zeros((action_length, 7))
    new_action = np.concatenate((action, padding_action), axis=0)
    return new_action

def match_actions_to_language(actions, language_prediction_length, action_length):
    '''
        Now we only use direciton, ignoring the rotation
        We check the actions in 5 timesteps, and return the direction of the action and gripper action
        action shape is (T, 7)
    '''
    
    def sum_action_to_language(direction, gripper):

        # if gripper[0] > 0 and gripper[-1] < 0:
        #     gripper_language = "I need to open the gripper."
        #     return gripper_language
        # elif gripper[0] < 0 and gripper[-1] > 0:
        #     gripper_language = "I need to close the gripper."
        #     return gripper_language
        
        if gripper[0] < 0:
            gripper_language = "with gripper open."
        else:
            gripper_language = "with gripper closed."
        
        language_prompt = "I need to move arm {} {}"
        
        direction_abs = abs(direction)
        action_threshold = 0.4
        
        direction_terms = {
            0: ('forward', 'backward'),
            1: ('right', 'left'),
            2: ('upward', 'downward')
        }
        
        def get_direction_language(idx, direction):
            primary = direction_terms[idx[0]]
            secondary = direction_terms[idx[1]]
            
            # Determine primary direction
            primary_direction = primary[0] if direction[idx[0]] > 0 else primary[1]
            
            # Determine secondary direction
            secondary_direction = secondary[0] if direction[idx[1]] > 0 else secondary[1]
            
            l = [primary_direction, secondary_direction]
            l = sorted(l)
            # Combine directions into a language string
            return f"{l[0]} and {l[1]}"
        
        direction_tf = [k > action_threshold for k in direction_abs]
        if any(direction_tf):
            if sum(direction_tf) == 1:
                idx = direction_tf.index(True)
                if direction[idx] > 0:
                    arm_language = f"I need to move arm {direction_terms[idx][0]} {gripper_language}"
                else:
                    arm_language = f"I need to move arm {direction_terms[idx][1]} {gripper_language}"
                return arm_language
            else:
                direciton_idxs = np.argsort(abs(direction))[::-1]
                direction_language = get_direction_language(direciton_idxs, direction)
                return language_prompt.format(direction_language, gripper_language)
        
        else:
            arm_language = "I need to make slight adjustments to the position of the end effector."

            return arm_language
        
        
    action_list = [actions[i: i+action_length] for i in range(len(actions)-action_length)]
    language_list = []
    for action_set  in action_list:
        direction = action_set[:language_prediction_length, :3]
        rotation = action_set[:language_prediction_length,3:6]
        gripper = action_set[:language_prediction_length,6]

        future_action = sum(direction)
        
        languages = sum_action_to_language(future_action, gripper)
        # languages = "make coffee"
        
        language_list.append(languages)

    return action_list, language_list


TASK_MAPPING = {
    "coffee_d0": "make coffee",
    # "coffee_d1": "make coffee",
    # "stack_d0": "stack the red block on top of the green block",
    # "stack_d1": "stack the red block on top of the green block",
    # "stack_three_d0": "stack the blocks in the order of blue, red, and green from top to bottom",
    # "stack_three_d1": "stack the blocks in the order of blue, red, and green from top to bottom",
    # "threading_d0": "insert the needle into the needle hole",
    # "three_piece_assembly_d0":"stack the three pieces",
    # "three_piece_assembly_d1":"stack the three pieces",
}

if __name__ == "__main__":
    
    for key in TASK_MAPPING:
        data_path = "./origin_datasets/{}.hdf5".format(key)
        save_path = "./adjust_llava_motion/{}_adjust_llava_motion.hdf5".format(key)

        data = h5py.File(data_path,"r")
        save_data = h5py.File(save_path,"a")

        data_group = save_data.create_group("data")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        action_length = 4
        language_prediction_length = 4

        for i in range(501):
            action_data = data['data'][f'demo_{i}']['actions']
            action_data = np.array(action_data)
            data_group.create_group(f"demo_{i}")
            filled_action = fill_action(action_data, action_length)
            action_list, language_list = match_actions_to_language(filled_action,language_prediction_length, action_length)
            print("the language_list is:", language_list[0])
            language_token = clip.tokenize(language_list).to(device)

            with torch.no_grad():
                text_features = model.encode_text(language_token)
                print("the language feature shape is:", text_features.shape)
                
            language_feature = text_features.cpu().numpy()
            
            save_data['data'][f'demo_{i}']['action_chunking'] = action_list
            save_data['data'][f'demo_{i}']['language'] = language_list
            save_data['data'][f'demo_{i}']['language_feature'] = language_feature
            print("i is:", i)
            # saved_data = convert_dataset(action_list, language_list, data,"test.h5py")
    
    
    
