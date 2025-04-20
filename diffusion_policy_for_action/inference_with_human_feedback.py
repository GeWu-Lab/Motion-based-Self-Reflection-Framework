import warnings
import os
import hydra
import numpy as np
import torch
import random
from torch.utils.tensorboard import SummaryWriter
import json
import re
import policies.transformer_policy
import time
import utils
import trajectory_data
import wandb
from omegaconf import OmegaConf
from easydict import EasyDict
import yaml
from robomimic.config.config import Config
import os
# os.environ['MUJOCO_GL'] = 'osmesa'
# os.environ['DISPLAY'] = "0"
import cv2
import llava
import numpy as np
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

import requests
from PIL import Image
from io import BytesIO

from robosuite.controllers import load_controller_config
import robosuite as suite
import mimicgen_envs
import imageio

from PIL import Image, ImageDraw, ImageFont

json_path = "train_datasets/adjust_llava_motion/language_idx.json"
json_data = json.load(open(json_path))['language_idx']
json_key = list(json_data.keys())

for key in json_key:
    value = json_data[key]
    json_data[value] = key

class WorkSpace:
    def __init__(self, cfg):
        self.work_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.cfg = cfg
        self.device = cfg.device

        self.loss_threshold = cfg.loss_threshold

        

        shape_meta = {
            'third_rgb' : (3,84,84)
        }

        print("the shape meta is:", shape_meta)
        self.policy = eval(cfg.policy._target_)(cfg.policy, shape_meta).to(self.device)

        self.augmentation = trajectory_data.construct_augmentation(cfg.dataset.kwargs.augmentation_cfg, self.cfg.policy.encoder.network_kwargs.name)

        self.sw = SummaryWriter(os.path.join(self.work_dir, 'tb'))



    def save_snapshot(self, idx):
        save_path = os.path.join(self.work_dir, f'snapshot_{idx}.pth')
        save_data = self.policy.state_dict()
        data = {
            "policy": save_data,
            "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg
        }
        with open(save_path,"wb") as f:
            torch.save(data,f)
    
    def load_snapshot(self, snapshot):
        with open(snapshot, "rb") as f:
            data = torch.load(f)
        
        weight = data['policy']
        new_weight = {}
        for weight_name in weight.keys():
            if "module." in weight_name:
                new_weight[weight_name.replace("module.","")] = weight[weight_name]
        
        self.policy.load_state_dict(new_weight)

    def map_tensor_to_device(self, data):
        """Move data to the device specified by self.cfg.device."""
        return utils.map_tensor(
            data, lambda x: utils.safe_device(x, device=self.cfg.device)
        )

    def to_torch(self, data, device):
        return torch.from_numpy(data).float().to(device)

    def load_llava_model(self):
        model_base = self.cfg.model_base
        model_path = self.cfg.model_path
        load_8bit = self.cfg.load_8bit
        load_4bit = self.cfg.load_4bit
        model_name = get_model_name_from_path(model_path)
        
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit, load_4bit, device=self.device)
        
        
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
        
        self.llava_model = model
        self.llava_tokenizer = tokenizer
        self.llava_image_processor = image_processor
        self.llava_context_len = context_len
        
        self.conv_mode = conv_mode
        self.roles = roles
    
    def llava_inference(self, image, task):
        image = Image.fromarray(image)
        image_tensor = process_images([image], self.llava_image_processor, self.llava_model.config)
        image_tensor = image_tensor.to(self.device, dtype=torch.float16)

        # first message
        if self.llava_model.config.mm_use_im_start_end:
            llava_language_input = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + task
        else:
            llava_language_input = DEFAULT_IMAGE_TOKEN + '\n' + task
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], llava_language_input)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, self.llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output_ids = self.llava_model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[self.cfg.image_shape],
                do_sample=True if self.cfg.temperature > 0 else False,
                temperature=self.cfg.temperature,
                max_new_tokens=self.cfg.max_new_tokens,
                # streamer=streamer,
                use_cache=True)
        outputs = self.llava_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return outputs
        # pattern = r"<s>(.*?)</s>"
        # result = re.search(pattern, outputs)
        # if result:
        #     extracted_text = result.group(1)
        # return extracted_text

    def make_env(self, env_name = "Coffee_D0"):
        self.options = {}
        self.options["env_name"] = env_name
        self.options["robots"] = "Panda"
        self.options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")
        
        self.env = suite.make(
            **self.options,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=True,
            control_freq=20,
            camera_heights=84,
            camera_widths=84,
            reward_shaping= False,
            camera_names= ["agentview","robot0_eye_in_hand"]
        )

    def preprocess_img(self,img):
        # img = np.flipud(img)

        img = img[::-1].copy()
        # img = img[:,:,::-1]
        # imageio.imsave("now.jpg", img)
        img = img/255.0
        img = img.transpose(2,0,1)
        
        # print("the img is:", img.shape)
        
        return img

    def draw_language(self, rgb, text):
        record_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(record_img)
        position = (100, 300)
        color = 'rgb(0, 0, 0)'  # 
        draw.text(position, text, fill=color)
        rgb = np.array(record_img)
        return rgb
    
    def inference(self):
        action_temporal_weights = np.exp(-1 * self.cfg.action_weight_init * np.arange(1, self.cfg.action_length + 1, 1))
        TASK_MAPPING = {
            "Coffee_D0": "make coffee",
            "Coffee_D1": "make coffee",
            # "Stack_D0": "stack the red block on top of the green block",
            # "Stack_D1": "stack the red block on top of the green block",
            # "StackThree_D0": "stack the blocks in the order of blue, red, and green from top to bottom",
            # "StackThree_D1": "stack the blocks in the order of blue, red, and green from top to bottom",
            # "Threading_D0": "insert the needle into the needle hole",
            # "Square_D0": "slide the square block onto the wooden stick",
            # "ThreePieceAssembly_D0":"stack the three pieces",
            # "ThreePieceAssembly_D1":"stack the three pieces"
        }
        # inference_env = ["Coffee_D0","Coffee_D1"]
        disable_torch_init()
        self.load_llava_model()
        import clip

        save_dir = "online_feedback"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        from video import VideoRecorder
        self.video_recorder = VideoRecorder(self.work_dir, name="third")
        self.video_recorder.init(enabled=True)
        self.ego_video_recorder = VideoRecorder(self.work_dir, name="ego")
        self.ego_video_recorder.init(enabled=True)
        terminate_on_success = True
        device = self.cfg.device
        model, preprocess = clip.load("ViT-B/32", device=self.cfg.device)
        
        self.policy.cuda()
        self.policy.eval()

        total_success_rate_dict = {}
        total_success_rate = []
        
        for env_name in TASK_MAPPING.keys():
            self.make_env(env_name)
            single_env_success_rate = []
        
            for rollout_time in range(self.cfg.rollout_time):
                predict_action_history = []
                success_flag = False
                ob_dict = None
                ob_dict = self.env.reset()
                third_obs = ob_dict['agentview_image']
                step_i = 0
                self.policy.reset()
                
                temporal_ensemble_history = []
                feedback_flag = False
                while step_i < self.cfg.inference_horizon:
                    llava_input_obs = self.env.sim.render(height=336, width=336, camera_name="agentview")
                    llava_input_obs = llava_input_obs[::-1]
                    
                    task_name = TASK_MAPPING[env_name]
                    language = self.llava_inference(image = llava_input_obs, task = f"Suppose you are the robot to {task_name}, what would you do next?")
                    print("the output language is:", language)
                    imageio.imsave(f"online_feedback/{env_name}.jpg", llava_input_obs)
                    
                    
                    if step_i % 60 == 0:
                        human_feedback = input("Input the flag human feedback:")
                        if "c" not in human_feedback:
                            feedback_flag = True
                            
                    if feedback_flag:
                        print("idx:", json_data)
                        human_feedback = input("Input the human motion feedback:")
                        if "c" in human_feedback:
                            feedback_flag = False
                            language_idx = torch.tensor(json_data[language])
                        else:
                            language_idx = torch.tensor(int(human_feedback))
                            corresponding_language = json_data[int(human_feedback)]
                            language += f"\n {corresponding_language}"
                    else:
                        language_idx = torch.tensor(json_data[language])
                    
                    
                    language_token = clip.tokenize([language]).to(device)
                    with torch.no_grad():
                        text_features = model.encode_text(language_token)
                    language_feature = text_features[0]
                    
                    
                    for infer_step in range(4):
                        third_obs = ob_dict['agentview_image']
                        ego_obs = ob_dict['robot0_eye_in_hand_image']
                        ee_position = ob_dict['robot0_eef_pos']
                        ee_quat = ob_dict['robot0_eef_quat']
                        gripper_state = ob_dict['robot0_gripper_qpos']
                        
                        state = np.concatenate([ee_position, ee_quat, gripper_state], axis=-1)
                        
                        third_obs = self.preprocess_img(third_obs)
                        ego_obs = self.preprocess_img(ego_obs)
                        third_obs = self.to_torch(third_obs,device).unsqueeze(0).unsqueeze(0)
                        ego_obs = self.to_torch(ego_obs,device).unsqueeze(0).unsqueeze(0)
                        state = self.to_torch(state,device).unsqueeze(0).unsqueeze(0)
                        language_feature = language_feature.unsqueeze(0).unsqueeze(0)
                        
                        obs = {
                            "third_rgb": third_obs,
                            "ego_rgb": ego_obs,
                            "language_feature": language_feature,
                            "states": state,
                            'language_idx': language_idx.unsqueeze(0).unsqueeze(0),
                        }

                        for key in obs.keys():
                            obs[key] = obs[key].to(self.cfg.device, non_blocking=True)
                            # use augmentation in gpu
                            if "rgb" in key or "depth" in key:
                                B,T,C,H,W = obs[key].shape
                                new_obs = obs[key].view(B*T,C,H,W)
                                aug_result = self.augmentation(new_obs)
                                obs[key] = aug_result.view(B,T,C,H,W)

                        if infer_step == 0:
                            actions = self.policy.get_action(obs)

                            action = actions.cpu().numpy()[0]
                            # if language_idx == 0:
                            #     action[:] = [0,0,0,0,0,0,1]
                            # elif language_idx == 19:
                            #      action[:] = [0,0,0,0,0,0,-1]
                        else:
                            self.policy.add_obs_history(obs)
                            
                        obs, r, done, _ = self.env.step(action[infer_step])
                        ob_dict = obs
                        # print("the ob dict is:", ob_dict['agentview_image'].shape)
                        # rgb = self.env.render("rgb_array",height=256,width=256)
                        im = self.env.sim.render(height=336, width=336, camera_name="agentview")
                        rgb = im[::-1]
                        text_rgb = self.draw_language(rgb, language)
                        self.video_recorder.record(text_rgb)
                        step_i += 1
                        success = r
                        
                        if done or (terminate_on_success and success):
                            success_flag = True
                            break
                        
                        
                        print("the step is:", step_i)
                        print("the rollout time is:", rollout_time)

                    if success_flag:
                        break

                single_env_success_rate.append(success)
                self.video_recorder.save(f"{env_name}_{rollout_time}.mp4")
                
            print("the {} task success rate is: {}".format(env_name, np.mean(single_env_success_rate)))
            total_success_rate_dict[env_name] = np.mean(single_env_success_rate)
            total_success_rate.append(np.mean(single_env_success_rate))
            
        total_success_rate_dict['mean'] = np.mean(total_success_rate)
        json_path = os.path.join(self.work_dir, 'success_rate.json')
        json.dump(total_success_rate_dict, open(json_path,"w"))

@hydra.main(config_path='cfgs', config_name='inference_config',version_base=None)
def main(cfg):
    # print(os.environ['LD_LIBRARY_PATH'])
    # print(os.environ['DISPLAY'])
    # print(os.environ['MUJOCO_GL'])
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    w = WorkSpace(cfg)

    if cfg.load_bc:
        print("load the behavior cloning model")
        w.load_snapshot(cfg.bc_path)
    w.inference()


if __name__ == '__main__':
	main()