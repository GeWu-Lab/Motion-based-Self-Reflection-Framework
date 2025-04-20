import warnings
import os
import hydra
import numpy as np
import torch
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
import clip
import torch.nn.functional as F

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from venv import MyInferenceEnv, VectorEnvs

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
language_keys = list(json_data.keys())
        


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

    def eval(self):
        losses = []
        
        for batch in self.eval_dataloader:
            obs = batch['obs']
            for key in obs.keys():
                a = time.time()
                obs[key] = obs[key].to(self.cfg.device, non_blocking=True)
                # use augmentation in gpu
                if "rgb" in key or "depth" in key:
                    # print("the key is:", key)
                    # print("the obs shape is:", obs[key].shape)
                    B,T,C,H,W = obs[key].shape
                    new_obs = obs[key].view(B*T,C,H,W)
                    aug_result = self.augmentation(new_obs)
                    obs[key] = aug_result.view(B,T,C,H,W)
            with torch.no_grad():
                gt_action = batch['actions'].to(self.cfg.device, non_blocking=True)
                loss = self.policy.compute_loss(obs, gt_action)
                losses.append(loss.item())
        return np.mean(losses)


    def train(self):
        train_step = 0
        best_eval_loss = 999999
        best_train_loss = 999999
        train_first_flag = True
        for epoch in range(self.cfg.epoch):

            train_losses = []
            for batch in self.train_dataloader:


                # batch = self.map_tensor_to_device(batch)
                start_time = time.time()
                train_step += 1
                obs = batch['obs']
                for key in obs.keys():
                    obs[key] = obs[key].to(self.cfg.device, non_blocking=True)
                    # use augmentation in gpu
                    if "rgb" in key or "depth" in key:
                        B,T,C,H,W = obs[key].shape
                        new_obs = obs[key].view(B*T,C,H,W)
                        aug_result = self.augmentation(new_obs)
                        obs[key] = aug_result.view(B,T,C,H,W)
                    
                gt_action = batch['actions'].to(self.cfg.device, non_blocking=True)

                self.optimizer.zero_grad()
                loss = self.policy.compute_loss(obs, gt_action)
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                end_time = time.time()
                duration = end_time - start_time
                print("the loss is:", loss)

            epoch_loss = np.mean(train_losses)
            
            if epoch_loss < best_train_loss:
                best_train_loss = epoch_loss
                self.save_snapshot("best_train")

            if train_first_flag and epoch_loss < self.loss_threshold:
                train_first_flag = False
                self.save_snapshot("first_train_threshold")
                
            if epoch % 5 == 0:
                print("i save the snapshot")
                self.save_snapshot(epoch)
                # eval_loss = self.eval()
                # if eval_loss < best_eval_loss:
                #     best_eval_loss = eval_loss
                #     self.save_snapshot("best")

                self.sw.add_scalar('Loss/train', loss, train_step)
                # self.run.log({"Loss/train": loss, "Duration/train": duration}, step=train_step)

            if epoch % 5 == 0:
                print("i save the snapshot")
                self.save_snapshot(epoch)
        self.sw.close()
        self.run.finish()

    def to_torch(self, data, device):
        return torch.from_numpy(data).float().to(device)

    def load_llava_model(self):
        model_base = self.cfg.model_base
        model_path = self.cfg.model_path
        load_8bit = self.cfg.load_8bit
        load_4bit = self.cfg.load_4bit
        model_name = get_model_name_from_path(model_path)
        
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit, load_4bit, device=self.device)
        
        for name, param in model.named_parameters():
            if "mm" in name:
                print(f"{name}: shape={param.shape}")
        
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
    
    def llava_inference(self, images, task):
        images = np.array(images)
        input_image = [Image.fromarray(image) for image in images]
        image_tensor = process_images(input_image, self.llava_image_processor, self.llava_model.config)
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
        print("the input ids shape is:", input_ids.shape)
        input_ids = input_ids.repeat(image_tensor.shape[0], 1)
        # import pdb; pdb.set_trace()
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
        output_languages = []
        for output_id in output_ids:
            output_language = self.llava_tokenizer.decode(output_id, skip_special_tokens=True).strip()
            output_languages.append(output_language)
        return output_languages
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
        
        env = suite.make(
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
        return env

    def preprocess_img(self,img):
        # img = np.flipud(img)

        img = img[:,::-1].copy()
        # img = img[:,:,::-1]
        # imageio.imsave("now.jpg", img)
        img = img/255.0
        img = img.transpose(0,3,1,2)
        
        # print("the img is:", img.shape)
        
        return img

    def draw_language(self, rgb, text):
        record_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(record_img)
        position = (0, 200)
        color = 'rgb(0, 0, 0)'  # 
        draw.text(position, text, fill=color)
        rgb = np.array(record_img)
        return rgb
    

    def search_nearest_idx(self,l,clip_model,predefined_text_features):
        motion_language_token = clip.tokenize([l]).to(self.cfg.device)
        with torch.no_grad():
            motion_text_features = clip_model.encode_text(motion_language_token)
        cosine_similarity = F.cosine_similarity(motion_text_features, predefined_text_features, dim=1)
        
        corresponding_idx = torch.argmax(cosine_similarity)

        return corresponding_idx
        
        
        
    
    def inference(self):
        action_temporal_weights = np.exp(-1 * self.cfg.action_weight_init * np.arange(1, self.cfg.action_length + 1, 1))
        TASK_MAPPING = {
            "Coffee_D0": "make coffee",
            "Coffee_D1": "make coffee",
            "Stack_D0": "stack the red block on top of the green block",
            "Stack_D1": "stack the red block on top of the green block",
            "StackThree_D0": "stack the blocks in the order of blue, red, and green from top to bottom",
            "StackThree_D1": "stack the blocks in the order of blue, red, and green from top to bottom",
            "Threading_D0": "insert the needle into the needle hole",
            # "Square_D0": "slide the square block onto the wooden stick",
            "ThreePieceAssembly_D0":"stack the three pieces",
            "ThreePieceAssembly_D1":"stack the three pieces",
            # "MugCleanup_D0":"put the mug into the drawer",
            # "MugCleanup_D1":"put the mug into the drawer",
            
            # "Object_Stack_D0": "stack the blue block on top of the green block"
            # "Opposed_MugCleanup_D0":"put the mug into the drawer",
        }
        # inference_env = ["Coffee_D0","Coffee_D1"]
        disable_torch_init()
        self.load_llava_model()
        

        terminate_on_success = True
        device = self.cfg.device
        model, preprocess = clip.load("ViT-B/32", device=self.cfg.device)
        
        language_list = language_keys
        predefined_language_token = clip.tokenize(language_list).to(device)
        
        with torch.no_grad():
            predefined_text_features = model.encode_text(predefined_language_token)
            print("the language feature shape is:", predefined_text_features.shape)
        
        
        self.policy.cuda()
        self.policy.eval()
        
        rollout_time = self.cfg.rollout_time
        for env_name in TASK_MAPPING.keys():
            multi_envs = []
            for _ in range(rollout_time):
                env = self.make_env(env_name)
                multi_envs.append(MyInferenceEnv(env, 10))
            
            inference_env = VectorEnvs(multi_envs, work_dir=self.work_dir, task_name = env_name)
        
            ob_dict = inference_env.reset(reset_all = True)
            step_i = 0
            self.policy.reset()
            reset_indexs = []

            
            while True:
                if step_i % self.cfg.action_length == 0:
                    llava_input_obs = inference_env.render()
                    
                    task_name = TASK_MAPPING[env_name]
                    language = self.llava_inference(images = llava_input_obs, task = f"Suppose you are the robot to {task_name}, what would you do next?")
                    print("the output language is:", language)
                    
                    for language_idx, l in enumerate(language):
                        if l not in language_keys:
                            cor_language_idx = self.search_nearest_idx(l,model, predefined_text_features)
                            language[language_idx] = language_keys[cor_language_idx]
                            
                    language_idxs = [json_data[l] for l in language]
                    language_idxs = self.to_torch(np.array(language_idxs),device).long()
                    language_token = clip.tokenize(language).to(device)
                    with torch.no_grad():
                        text_features = model.encode_text(language_token)
                    language_feature = text_features
                
                # print("the language feature is:", language_feature.shape)
                
                

                third_obs = ob_dict['agentview_image']
                ego_obs = ob_dict['robot0_eye_in_hand_image']
                ee_position = ob_dict['robot0_eef_pos']
                ee_quat = ob_dict['robot0_eef_quat']
                gripper_state = ob_dict['robot0_gripper_qpos']
                
                state = np.concatenate([ee_position, ee_quat, gripper_state], axis=-1)
                
                third_obs = self.preprocess_img(third_obs)
                ego_obs = self.preprocess_img(ego_obs)
                third_obs = self.to_torch(third_obs,device).unsqueeze(1)
                ego_obs = self.to_torch(ego_obs,device).unsqueeze(1)
                state = self.to_torch(state,device).unsqueeze(1)
                language_feature_input = language_feature.unsqueeze(1)
                
                obs = {
                    "third_rgb": third_obs,
                    "ego_rgb": ego_obs,
                    "language_feature": language_feature_input,
                    "language_idx": language_idxs.unsqueeze(1),
                    "states": state
                }

                for key in obs.keys():
                    obs[key] = obs[key].to(self.cfg.device, non_blocking=True)
                    # use augmentation in gpu
                    if "rgb" in key or "depth" in key:
                        B,T,C,H,W = obs[key].shape
                        new_obs = obs[key].view(B*T,C,H,W)
                        aug_result = self.augmentation(new_obs)
                        obs[key] = aug_result.view(B,T,C,H,W)
            
                if step_i % self.cfg.action_length == 0:
                    actions = self.policy.get_action(obs, reset_indexs)
                    actions = actions.cpu().numpy()
                    reset_indexs = []
                else:
                    self.policy.add_obs_history(obs)
                
                print(inference_env.envs[0].rollout_time)
                
                ob_dict, resets = inference_env.step(actions[:,step_i], language)
                reset_indexs = resets
                
                step_i = (step_i + 1 ) % self.cfg.action_length
                
                if step_i == 0:
                    ob_dict = inference_env.reset(reset_idxs = reset_indexs)

                    
                if inference_env.is_done():
                    print("the task is finished")
                    break

            inference_env.close()


@hydra.main(config_path='cfgs', config_name='inference_config',version_base=None)
def main(cfg):
    import random
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