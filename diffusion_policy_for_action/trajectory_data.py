import numpy as np
import torch
import os
import torch.nn as nn
from torchvision import transforms
from augmentation import *
import time
import h5py
import imageio

def construct_augmentation(augmentation_cfg, encoder = "resnet"):
    img_transforms = []
    # img_transforms.append(transforms.ToTensor())
    if encoder == "clip":
        print("the encoder is clip")
        img_transforms.append(transforms.Normalize(mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711]))
    elif encoder == "vit":
        print("the encoder is vit")
        img_transforms.append(transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]))
    else:
        print("the encoder is resnet")
        img_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    if "shift" in augmentation_cfg:
        img_transforms.append(RandomShiftsAug(pad=5))
    
    if "color" in augmentation_cfg:
        img_transforms.append(ImgColorJitterAug())
    
    if "flip" in augmentation_cfg:
        img_transforms.append(transforms.RandomHorizontalFlip())
    
    # print("the img_transforms is:", img_transforms)
    return transforms.Compose(img_transforms)

class Dataset(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def get_dataloader(self, num_workers = 6,shuffle=True, sampler = None):
        return torch.utils.data.DataLoader(self, batch_size=self.batch_size,sampler = sampler,num_workers=num_workers, drop_last=True, pin_memory=True)
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError

class SingleFrameDataset(Dataset):
    def __init__(self, data_dir, batch_size = 128, action_length=10, repeat_time = 10):
        super(SingleFrameDataset, self).__init__(batch_size)
        self.data_dir = data_dir
        self.action_length = action_length
        self.repeat_time = repeat_time
        
        state_action_path = os.path.join(data_dir, "expert_data.npz")
        state_action_data = np.load(state_action_path, allow_pickle=True)
        self.states = state_action_data["states"]
        self.actions = state_action_data["actions"]
        
        img_dir = os.path.join(data_dir, "images")
        img_paths = []
        data_names = os.listdir(img_dir)
        data_names = sorted(data_names)
        
        for idx,name in enumerate(data_names):
            img_path = os.path.join(img_dir, name)
            img_paths.append(img_path)
        
        self.img_paths = img_paths
        self.data_len = len(self.states)        
    
    def __len__(self):
        return self.data_len * self.repeat_time
    
    def __getitem__(self, index):
        true_idx = index % self.data_len
        states = self.states[true_idx]
        actions = self.actions[true_idx]
        img_path = self.img_paths[true_idx]
        img = np.load(img_path)["data"]
        
        traj_len = len(states)
        start_idx = np.random.randint(0, traj_len - self.action_length)
        end_idx = start_idx + self.action_length
        
        data = {
            "obs":{
                "visual": img[start_idx],
                "states": states[start_idx],
            },
            "actions": actions[start_idx:end_idx],
        }

        return data

def get_full_imgs(data_dir, dir_name, is_depth=False):
    img_dir = os.path.join(data_dir, dir_name)
    img_paths = []
    data_names = os.listdir(img_dir)
    data_names = sorted(data_names)
    
    for idx,name in enumerate(data_names):
        img_path = os.path.join(img_dir, name)
        img_paths.append(img_path)
    
    imgs = []
    
    for img_path in img_paths:
        img = np.load(img_path)["data"]
        # change bgr to rgb
        # img = img[:, :, :, ::-1]
        if is_depth:
            img = np.expand_dims(img, axis=1)
            img = np.repeat(img, 3, axis=1)
        else:
            img = img[..., ::-1]
            img = img.transpose(0,3,1,2) / 255.0
        # img = torch.from_numpy(img).float()
        # shape of img: (TimeSteps, C, H, W)
        imgs.append(img)
        # shape of imgs: (Rollouts, TimeSteps, C, H, W)
    return imgs   

def get_imgs(img_path, is_depth=False):
    img = np.load(img_path)["data"]
        # change bgr to rgb
        # img = img[:, :, :, ::-1]
    if is_depth:
        img = np.expand_dims(img, axis=1)
        img = np.repeat(img, 3, axis=1)
    else:
        img = img[..., ::-1]
        img = img.transpose(0,3,1,2) / 255.0
    # img = torch.from_numpy(img).float()
    return np.array(img)


def aug_imgs(img, is_depth=False):
    if is_depth:
        img = np.expand_dims(img, axis=1)
        img = np.repeat(img, 3, axis=1)
    else:
        img = img[..., ::-1]
        img = img.transpose(0,3,1,2) / 255.0
    # img = torch.from_numpy(img).float()
    return np.array(img)

def get_imgs_path(data_dir, dir_name, is_depth=False):
    img_dir = os.path.join(data_dir, dir_name)
    img_paths = []
    data_names = os.listdir(img_dir)
    data_names = sorted(data_names)
    
    for idx,name in enumerate(data_names):
        img_path = os.path.join(img_dir, name)
        img_paths.append(img_path)
    return img_paths

import json

class TrajectoryLoadDataset(Dataset):
    def __init__(self, data_dir, data_path, language_condition, batch_size = 128, frame_length=5, demonstration_num = 10,augmentation_cfg = [""], have_depth=False, have_ego=True):
        super(TrajectoryLoadDataset, self).__init__(batch_size)
        self.data_path = data_path
        self.frame_length = frame_length
        
        annotation_path = os.path.join(data_dir, data_path)
        print("the annotation path is:", annotation_path)
        with open(annotation_path,"r") as f:
            mapping_data = json.load(f)
            data_names = mapping_data['dataset']
        print("the data names are:", data_names)
        obs_data_set = []
        language_action_set = []
        
        for name in data_names:
            obs_data = h5py.File(os.path.join(data_dir, name + ".hdf5"), "r",swmr=True, libver='latest')['data']
            language_action_data = h5py.File(os.path.join(data_dir, name + "_" + language_condition + ".hdf5"), "r")['data']
            
            obs_data_set.append(obs_data)
            language_action_set.append(language_action_data)
        
        index_map = []
        # self.augmentation = construct_augmentation(augmentation_cfg) 
        
        third_rgb_data = []
        ego_rgb_data = []
        action_chunking_data = []
        states_data = []
        language_data = []
        
        total_idx = 0
        for idx in range(len(data_names)):
            demo_n = 0
            traj_data = obs_data_set[idx]
            language_action_data = language_action_set[idx]
            # print("the length is:", len(traj_data))
            for name in traj_data.keys():
                demonstration_traj = traj_data[name]
                language_action_traj = language_action_data[name]
                
                img_obs = np.array(demonstration_traj["obs"]['agentview_image'])
                ego_obs = np.array(demonstration_traj["obs"]['robot0_eye_in_hand_image'])

                # img_obs = self.preprocess_img(img_obs)
                # ego_obs = self.preprocess_img(ego_obs)

                ee_pos = demonstration_traj['obs']['robot0_eef_pos']
                ee_quat = demonstration_traj['obs']['robot0_eef_quat']
                gripper = demonstration_traj['obs']['robot0_gripper_qpos']
                
                language = np.array(language_action_traj['language_feature'])
                action = np.array(language_action_traj['action_chunking'])
                
                third_rgb_data.append(img_obs)
                ego_rgb_data.append(ego_obs)
                state = np.concatenate([ee_pos, ee_quat, gripper], axis=-1)
                states_data.append(state)
                language_data.append(language)
                action_chunking_data.append(action)
                
                obs_len = img_obs.shape[0]
                for img_idx, img in enumerate(img_obs):
                    if img_idx < obs_len and img_idx - frame_length >= 0:
                        index_map.append([total_idx, img_idx])
                
                total_idx += 1
                demo_n += 1
                if demo_n >= demonstration_num:
                    break
                print("the total idx is:", total_idx)

        # self.obs_data_set = obs_data_set
        # self.language_action_set = language_action_set
        self.index_map = index_map

        self.third_rgb_data = third_rgb_data
        self.ego_rgb_data = ego_rgb_data
        self.action_chunking_data = action_chunking_data
        self.states_data = states_data
        self.language_data = language_data

        self.data_len = len(self.index_map)  
                
        self.have_depth = have_depth
        self.have_ego = have_ego
        
        self.img_keys = ["third_rgb"]
        if self.have_depth:
            self.img_keys.append("third_depth")
        if self.have_ego:
            self.img_keys.append("ego_rgb")
            if self.have_depth:
                self.img_keys.append("ego_depth")

    def __len__(self):
        return self.data_len
    
    def get_shape_meta(self):

        data_shape = {}
        for key in self.img_keys:
            data_shape[key] = (3,84,84)
        
        return data_shape
    
    def fill_data(self, traj_num, start_idx, end_idx):
        return_dict = {}
        for key in self.img_keys:
            
            imgs_path = getattr(self, key)[traj_num]
            # print("the imgs path is:", imgs_path)
            imgs = h5py.File(imgs_path, "r")["data"]
            
            # print("the imgs shape is:", imgs.shape)
            return_visual_obs = imgs[start_idx:end_idx]
            return_visual_obs = aug_imgs(return_visual_obs, is_depth=(key.find("depth") != -1))
            
            if len(return_visual_obs) < self.frame_length:

                fill_visual_obs = np.empty((self.frame_length,) + return_visual_obs.shape[1:], dtype=np.float32)
                fill_visual_obs[:len(return_visual_obs)] = return_visual_obs
                fill_visual_obs[len(return_visual_obs):] = return_visual_obs[-1]
            
                return_dict[key] = torch.from_numpy(fill_visual_obs).float()
            else:
                return_dict[key] = torch.from_numpy(return_visual_obs).float()
        return return_dict
    
    def preprocess_img(self, img):
        img = img / 255.0
        # print(img.shape)
        img = np.transpose(img, (0,3,1,2))
        return img
    
    def to_torch(self, data):
        return torch.from_numpy(data).float()
    
    def __getitem__(self, index):
        
        # a = time.time()

        index_info = self.index_map[index]
        demo_idx = index_info[0]
        timestep_idx = index_info[1]

        # obs_data = self.obs_data_set[demo_idx][traj_num]
        # language_action = self.language_action_set[demo_idx][traj_num]

        # b = time.time()
        # data = self.traj_data[traj_num]
        # self.third_rgb_data = third_rgb_data
        # self.ego_rgb_data = ego_rgb_data
        # self.action_chunking_data = action_chunking_data
        # self.states_data = states_data
        # self.language_data = language_data
        
        third_img_obs = self.third_rgb_data[demo_idx][timestep_idx - self.frame_length:timestep_idx]
        ego_img_obs = self.ego_rgb_data[demo_idx][timestep_idx - self.frame_length:timestep_idx]
        
        action_chunking = self.action_chunking_data[demo_idx][timestep_idx - self.frame_length :timestep_idx]
        language_feature = self.language_data[demo_idx][timestep_idx - self.frame_length :timestep_idx]
        state = self.states_data[demo_idx][timestep_idx - self.frame_length :timestep_idx]

        # ee_pos = obs_data['obs']['robot0_eef_pos'][timestep_idx - self.frame_length:timestep_idx]
        # ee_quat = obs_data['obs']['robot0_eef_quat'][timestep_idx - self.frame_length:timestep_idx]
        # gripper = obs_data['obs']['robot0_gripper_qpos'][timestep_idx - self.frame_length:timestep_idx]

        # state = np.concatenate([ee_pos, ee_quat, gripper], axis=-1)

        # c = time.time()
        
        third_img_obs = self.preprocess_img(third_img_obs)
        ego_img_obs = self.preprocess_img(ego_img_obs)
        
        # third_img= self.augmentation(self.to_torch(third_img_obs))
        # ego_img = self.augmentation(self.to_torch(ego_img_obs))

        third_img= self.to_torch(third_img_obs)
        ego_img = self.to_torch(ego_img_obs)
        # d = time.time()
        
        # print("the action chunking shape is:", action_chunking.shape)
        action_chunking = action_chunking
        data = {
            "obs":{
                "third_rgb": third_img,
                "ego_rgb": ego_img,
                "language_feature": self.to_torch(language_feature),
                "states": self.to_torch(state)
            },
            "actions": self.to_torch(action_chunking),
        }
        # e = time.time()

        return data


class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, data_path, language_condition, batch_size = 128, frame_length=5, demonstration_num = 10,augmentation_cfg = [""], have_depth=False, have_ego=True):
        super(TrajectoryDataset, self).__init__(batch_size)
        self.data_path = data_path
        self.frame_length = frame_length
        
        annotation_path = os.path.join(data_dir, data_path)
        print("the annotation path is:", annotation_path)
        with open(annotation_path,"r") as f:
            mapping_data = json.load(f)
            data_names = mapping_data['dataset']
        print("the data names are:", data_names)
        obs_data_set = []
        language_action_set = []
        
        for name in data_names:
            obs_data = h5py.File(os.path.join(data_dir, name + ".hdf5"), "r",swmr=True, libver='latest')['data']
            language_action_data = h5py.File(os.path.join(data_dir, name + "_" + language_condition + ".hdf5"), "r")['data']
            
            obs_data_set.append(obs_data)
            language_action_set.append(language_action_data)
        
        index_map = []
        # self.augmentation = construct_augmentation(augmentation_cfg) 
        
        
        for idx in range(len(data_names)):
            demo_n = 0
            traj_data = obs_data_set[idx]
            print("the length is:", len(traj_data))
            for name in traj_data.keys():
                demonstration_traj = traj_data[name]

                img_obs = demonstration_traj["obs"]['agentview_image']
                obs_len = img_obs.shape[0]
                for img_idx in range(obs_len):
                    if img_idx < obs_len and img_idx - frame_length >= 0:
                        index_map.append([idx, name, img_idx])
                demo_n += 1
                if demo_n >= demonstration_num:
                    break

        self.obs_data_set = obs_data_set
        self.language_action_set = language_action_set
        self.index_map = index_map

        self.data_len = len(self.index_map)  
                
        self.have_depth = have_depth
        self.have_ego = have_ego
        
        self.img_keys = ["third_rgb"]
        if self.have_depth:
            self.img_keys.append("third_depth")
        if self.have_ego:
            self.img_keys.append("ego_rgb")
            if self.have_depth:
                self.img_keys.append("ego_depth")

    def __len__(self):
        return self.data_len
    
    def get_shape_meta(self):

        data_shape = {}
        for key in self.img_keys:
            data_shape[key] = (3,84,84)
        
        return data_shape
    
    def fill_data(self, traj_num, start_idx, end_idx):
        return_dict = {}
        for key in self.img_keys:
            
            imgs_path = getattr(self, key)[traj_num]
            # print("the imgs path is:", imgs_path)
            imgs = h5py.File(imgs_path, "r")["data"]
            
            # print("the imgs shape is:", imgs.shape)
            return_visual_obs = imgs[start_idx:end_idx]
            return_visual_obs = aug_imgs(return_visual_obs, is_depth=(key.find("depth") != -1))
            
            if len(return_visual_obs) < self.frame_length:

                fill_visual_obs = np.empty((self.frame_length,) + return_visual_obs.shape[1:], dtype=np.float32)
                fill_visual_obs[:len(return_visual_obs)] = return_visual_obs
                fill_visual_obs[len(return_visual_obs):] = return_visual_obs[-1]
            
                return_dict[key] = torch.from_numpy(fill_visual_obs).float()
            else:
                return_dict[key] = torch.from_numpy(return_visual_obs).float()
        return return_dict
    
    def preprocess_img(self, img):
        img = img / 255.0
        # print(img.shape)
        img = np.transpose(img, (0,3,1,2))
        return img
    
    def to_torch(self, data):
        return torch.from_numpy(data).float()
    
    def __getitem__(self, index):
        
        # a = time.time()

        index_info = self.index_map[index]
        demo_idx = index_info[0]
        traj_num = index_info[1]
        timestep_idx = index_info[2]

        obs_data = self.obs_data_set[demo_idx][traj_num]
        language_action = self.language_action_set[demo_idx][traj_num]

        # b = time.time()
        # data = self.traj_data[traj_num]
        third_img_obs = obs_data['obs']['agentview_image'][timestep_idx - self.frame_length:timestep_idx]
        ego_img_obs = obs_data['obs']['robot0_eye_in_hand_image'][timestep_idx - self.frame_length:timestep_idx]
        
        action_chunking = language_action['action_chunking'][timestep_idx - self.frame_length :timestep_idx]
        language_feature = language_action['language_feature'][timestep_idx - self.frame_length :timestep_idx]

        ee_pos = obs_data['obs']['robot0_eef_pos'][timestep_idx - self.frame_length:timestep_idx]
        ee_quat = obs_data['obs']['robot0_eef_quat'][timestep_idx - self.frame_length:timestep_idx]
        gripper = obs_data['obs']['robot0_gripper_qpos'][timestep_idx - self.frame_length:timestep_idx]

        state = np.concatenate([ee_pos, ee_quat, gripper], axis=-1)

        # c = time.time()
        
        third_img_obs = self.preprocess_img(third_img_obs)
        ego_img_obs = self.preprocess_img(ego_img_obs)
        
        # third_img= self.augmentation(self.to_torch(third_img_obs))
        # ego_img = self.augmentation(self.to_torch(ego_img_obs))

        third_img= self.to_torch(third_img_obs)
        ego_img = self.to_torch(ego_img_obs)
        # d = time.time()
        
        data = {
            "obs":{
                "third_rgb": third_img,
                "ego_rgb": ego_img,
                "language_feature": self.to_torch(language_feature),
                "states": self.to_torch(state)
            },
            "actions": self.to_torch(action_chunking),
        }
        # e = time.time()

        return data




class TrajectorySpeedDataset(Dataset):
    def __init__(self, data_dir, data_path, language_condition, batch_size = 128, frame_length=5, demonstration_num = 10,augmentation_cfg = [""], have_depth=False, have_ego=True):
        super(TrajectorySpeedDataset, self).__init__(batch_size)
        self.data_path = data_path
        self.frame_length = frame_length
        
        annotation_path = os.path.join(data_dir, data_path)
        print("the annotation path is:", annotation_path)
        with open(annotation_path,"r") as f:
            mapping_data = json.load(f)
            data_names = mapping_data['dataset']
        print("the data names are:", data_names)
        obs_data_set = []
        
        for name in data_names:
            obs_data = h5py.File(os.path.join(data_dir, name + "_" + language_condition + ".hdf5"), "r")['data']
            obs_data_set.append(obs_data)
        
        index_map = []
        # self.augmentation = construct_augmentation(augmentation_cfg) 
        
        for idx in range(len(data_names)):
            demo_n = 0
            traj_data = obs_data_set[idx]
            print("the length is:", len(traj_data))
            for name in traj_data.keys():
                demonstration_traj = traj_data[name]

                obs_len = len(demonstration_traj)
                for img_idx in range(obs_len):
                    if img_idx < obs_len and img_idx - frame_length >= 0:
                        index_map.append([idx, name, img_idx])
                demo_n += 1
                if demo_n >= demonstration_num:
                    break

        self.obs_data_set = obs_data_set

        self.index_map = index_map

        self.data_len = len(self.index_map)  
                
        self.have_depth = have_depth
        self.have_ego = have_ego
        
        self.img_keys = ["third_rgb"]
        if self.have_depth:
            self.img_keys.append("third_depth")
        if self.have_ego:
            self.img_keys.append("ego_rgb")
            if self.have_depth:
                self.img_keys.append("ego_depth")

    def __len__(self):
        return self.data_len
    
    def get_shape_meta(self):

        data_shape = {}
        for key in self.img_keys:
            data_shape[key] = (3,84,84)
        
        return data_shape
    
    def preprocess_img(self, img):
        img = img / 255.0
        # print(img.shape)
        img = np.transpose(img, (0,3,1,2))
        return img
    
    def to_torch(self, data):
        return torch.from_numpy(data).float()
    
    def __getitem__(self, index):
        index_info = self.index_map[index]
        demo_idx = index_info[0]
        traj_num = index_info[1]
        timestep_idx = index_info[2]

        obs_data = self.obs_data_set[demo_idx][traj_num]

        # Efficiently load data using list comprehension
        timesteps = range(timestep_idx - self.frame_length, timestep_idx)
        
        third_obs_set = np.array([obs_data[str(k)]['agentview_image'][...] for k in timesteps])
        ego_obs_set = np.array([obs_data[str(k)]['ego_image'][...] for k in timesteps])
        language_set = np.array([obs_data[str(k)]['language_feature'][...] for k in timesteps])
        action_set = np.array([obs_data[str(k)]['action_chunking'][...] for k in timesteps])
        state_set = np.array([
            np.concatenate([
                obs_data[str(k)]['ee_pos'][...], 
                obs_data[str(k)]['ee_quat'][...], 
                obs_data[str(k)]['gripper'][...]
            ], axis=-1) for k in timesteps
        ])

        third_img_obs = self.preprocess_img(third_obs_set)
        ego_img_obs = self.preprocess_img(ego_obs_set)
        
        third_img = self.to_torch(third_img_obs)
        ego_img = self.to_torch(ego_img_obs)

        data = {
            "obs": {
                "third_rgb": third_img,
                "ego_rgb": ego_img,
                "language_feature": self.to_torch(language_set),
                "states": self.to_torch(state_set)
            },
            "actions": self.to_torch(action_set),
        }
        
        return data
    

class TrajectoryIndexSpeedDataset(Dataset):
    def __init__(self, data_dir, data_path, language_condition, batch_size = 128, frame_length=5, demonstration_num = 10,augmentation_cfg = [""], have_depth=False, have_ego=True):
        super(TrajectoryIndexSpeedDataset, self).__init__(batch_size)
        self.data_path = data_path
        self.frame_length = frame_length
        
        annotation_path = os.path.join(data_dir, data_path)
        print("the annotation path is:", annotation_path)
        with open(annotation_path,"r") as f:
            mapping_data = json.load(f)
            data_names = mapping_data['dataset']
        print("the data names are:", data_names)
        obs_data_set = []
        
        for name in data_names:
            obs_data = h5py.File(os.path.join(data_dir, name + "_" + language_condition + ".hdf5"), "r")['data']
            obs_data_set.append(obs_data)
        
        index_map = []
        # self.augmentation = construct_augmentation(augmentation_cfg) 
        
        for idx in range(len(data_names)):
            demo_n = 0
            traj_data = obs_data_set[idx]
            print("the length is:", len(traj_data))
            for name in traj_data.keys():
                demonstration_traj = traj_data[name]

                obs_len = len(demonstration_traj)
                for img_idx in range(obs_len):
                    if img_idx < obs_len and img_idx - frame_length >= 0:
                        index_map.append([idx, name, img_idx])
                demo_n += 1
                if demo_n >= demonstration_num:
                    break

        self.obs_data_set = obs_data_set

        self.index_map = index_map

        self.data_len = len(self.index_map)  
                
        self.have_depth = have_depth
        self.have_ego = have_ego
        
        self.img_keys = ["third_rgb"]
        if self.have_depth:
            self.img_keys.append("third_depth")
        if self.have_ego:
            self.img_keys.append("ego_rgb")
            if self.have_depth:
                self.img_keys.append("ego_depth")

    def __len__(self):
        return self.data_len
    
    def get_shape_meta(self):

        data_shape = {}
        for key in self.img_keys:
            data_shape[key] = (3,84,84)
        
        return data_shape
    
    def preprocess_img(self, img):
        img = img / 255.0
        # print(img.shape)
        img = np.transpose(img, (0,3,1,2))
        return img
    
    def to_torch(self, data):
        return torch.from_numpy(data).float()
    
    def __getitem__(self, index):
        index_info = self.index_map[index]
        demo_idx = index_info[0]
        traj_num = index_info[1]
        timestep_idx = index_info[2]

        obs_data = self.obs_data_set[demo_idx][traj_num]

        # Efficiently load data using list comprehension
        timesteps = range(timestep_idx - self.frame_length, timestep_idx)
        
        third_obs_set = np.array([obs_data[str(k)]['agentview_image'][...] for k in timesteps])
        ego_obs_set = np.array([obs_data[str(k)]['ego_image'][...] for k in timesteps])
        language_idx_set = np.array([obs_data[str(k)]['language_idx'][...] for k in timesteps])
        action_set = np.array([obs_data[str(k)]['action_chunking'][...] for k in timesteps])
        state_set = np.array([
            np.concatenate([
                obs_data[str(k)]['ee_pos'][...], 
                obs_data[str(k)]['ee_quat'][...], 
                obs_data[str(k)]['gripper'][...]
            ], axis=-1) for k in timesteps
        ])
        language_set = np.array([obs_data[str(k)]['language_feature'][...] for k in timesteps])
        third_img_obs = self.preprocess_img(third_obs_set)
        ego_img_obs = self.preprocess_img(ego_obs_set)
        
        third_img = self.to_torch(third_img_obs)
        ego_img = self.to_torch(ego_img_obs)

        data = {
            "obs": {
                "third_rgb": third_img,
                "ego_rgb": ego_img,
                "language_idx": self.to_torch(language_idx_set).long(),
                "language_feature": self.to_torch(language_set),
                "states": self.to_torch(state_set)
            },
            "actions": self.to_torch(action_set),
        }
        
        return data

import clip
class MotionPredicetionDataset(Dataset):
    def __init__(self, batch_size,image_dir, json_path, motion_map_path, augmentation_cfg = [""]):
        super(MotionPredicetionDataset, self).__init__(batch_size)
        with open(json_path, "r") as f:
            json_data = json.load(f)
        with open(motion_map_path, "r") as f:
            motion_map = json.load(f)['language_idx']
        self.json_data = json_data
        
        
        print("the length of dataset is:", len(self.json_data))
        self.motion_map = motion_map
        # print(self.motion_map)
        # self.augmentation = construct_augmentation(augmentation_cfg) 
        self.image_dir = image_dir
    
    def __len__(self):
        return len(self.json_data)

    # def get_shape_meta(self):


    #     data_shape[key] = (3,336,336)
        
    #     return data_shape

    def __getitem__(self, idx):
        # print(idx)
        json_data = self.json_data[idx]
        image_path = json_data['image']
        image_path = os.path.join(self.image_dir, image_path)
        image = imageio.imread(image_path) / 255.0
        image = np.transpose(image, (2,0,1))
        image = image.astype(np.float32)
        query = json_data['conversations'][0]['value']
        
        answer = json_data['conversations'][1]['value']
        true_query = query.split(",")[0].split("the robot to ")[-1]
        language_token = clip.tokenize(true_query)
        
        answer_idx = self.motion_map[answer]
        
        return_dict = {
            "rgb_image": image,
            "query": language_token,
            "motion_idx": answer_idx
        }
        return return_dict


class MotionPredicetionNumpyDataset(Dataset):
    def __init__(self, batch_size,image_dir, json_path, numpy_path, augmentation_cfg = [""]):
        super(MotionPredicetionNumpyDataset, self).__init__(batch_size)
        with open(json_path, "r") as f:
            json_data = json.load(f)
        self.numpy_data = np.load(numpy_path,allow_pickle=True)
        self.json_data = json_data
        
        
        print("the length of dataset is:", len(self.json_data))
        # print(self.motion_map)
        self.image_dir = image_dir
    
    def __len__(self):
        return len(self.json_data)


    def __getitem__(self, idx):
        # print(idx)
        json_data = self.json_data[idx]
        image_path = json_data['image']
        image_path = os.path.join(self.image_dir, image_path)
        image = imageio.imread(image_path) / 255.0
        image = np.transpose(image, (2,0,1))
        image = image.astype(np.float32)
        
        language_token = self.numpy_data[idx]['language_token']
        answer_idx = self.numpy_data[idx]['motion_idx']
        return_dict = {
            "rgb_image": image,
            "query": language_token,
            "motion_idx": answer_idx
        }
        return return_dict

if __name__ == "__main__":
    dataset = TrajectoryDataset(data_path="./dataset/coffee_language_data.hdf5", batch_size=128, frame_length=5)
    loader = dataset.get_dataloader()
    for data in loader:
        print(data['actions'].shape)
        

