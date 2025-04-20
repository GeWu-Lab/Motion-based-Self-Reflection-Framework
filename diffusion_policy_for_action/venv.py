import numpy as np

from PIL import Image, ImageDraw, ImageFont
import os
import json

class MyInferenceEnv():
    def __init__(self, env,interact_time=10, timeout = 500):
        self.env = env
        self.rollout_time = -1
        self.all_finished = False
        self.rollout_finished = False
        self.interact_time = interact_time
        self.success_time = 0
        self.timeout = timeout
        self.timestep = 0
        self.image_list = []
        self.rollout_images = []
        self.success_flag = False
        
        self.language_set = []
        self.rollout_language = []
        
    def reset(self):
        self.success_flag = False
        if len(self.rollout_images) > 0:
            self.image_list.append(self.rollout_images)
            self.language_set.append(self.rollout_language)
        self.rollout_images = []
        self.rollout_language = []
        self.timestep = 0
        self.rollout_time += 1
        
        ob_dict = self.env.reset()
        return ob_dict
    
    def render(self, height=336, width=336, camera_name="agentview"):
        im = self.env.sim.render(height = height, width = width, camera_name = camera_name)
        rgb = im[::-1]
        return rgb
    
    def draw_language(self, rgb, text):
        record_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(record_img)
        position = (0, 280)
        color = 'rgb(0, 0, 0)'  # 
        draw.text(position, text, fill=color)
        rgb = np.array(record_img)
        return rgb
    
    def step_without_draw(self, action, language = "padding"):
        
        self.rollout_language.append(language)
        ob_dict, r, done, _ = self.env.step(action)
        rgb = self.render()
        
        # self.video_recorder.record(rgb)
        # self.image_list.append(rgb)
        # rgb = self.draw_language(rgb, language)
        
        self.rollout_images.append(rgb)
        success = r
        self.timestep += 1
        reset_flag = False
        
        if self.rollout_time >= self.interact_time:
            self.all_finished = True
            reset_flag = False
            return ob_dict, reset_flag
        
        if success:
            if self.success_flag is False:
                self.success_time += 1
                self.success_flag = True
            # ob_dict = self.env.reset()
            # self.image_list.append(self.rollout_images)
            
            reset_flag = True
        elif done or self.timestep > self.timeout:
            # ob_dict = self.env.reset()
            # self.image_list.append(self.rollout_images)
            reset_flag = True
            
        return ob_dict, reset_flag


    def step(self, action, language = "padding"):
        ob_dict, r, done, _ = self.env.step(action)
        rgb = self.render()
        
        # self.video_recorder.record(rgb)
        # self.image_list.append(rgb)
        rgb = self.draw_language(rgb, language)
        self.rollout_images.append(rgb)
        success = r
        self.timestep += 1
        reset_flag = False
        
        if self.rollout_time >= self.interact_time:
            self.all_finished = True
            reset_flag = False
            return ob_dict, reset_flag
        
        if success:
            if self.success_flag is False:
                self.success_time += 1
                self.success_flag = True
            # ob_dict = self.env.reset()
            # self.image_list.append(self.rollout_images)
            
            reset_flag = True
        elif done or self.timestep > self.timeout:
            # ob_dict = self.env.reset()
            # self.image_list.append(self.rollout_images)
            reset_flag = True
            
        return ob_dict, reset_flag

    def get_success_rate(self):
        return self.success_time / self.rollout_time

    def close(self):
        self.image_list = []
        self.rollout_images = []
        self.language_set = []
        self.rollout_language = []

def merge_dict(a,b):
    keys = ['agentview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos' ,'robot0_eef_quat', 'robot0_gripper_qpos']
    
    merged_dict_result = {}
    for key in keys:
        # print("the a[key] shape is:", a[key].shape)
        # print("the b[key] shape is:", b[key].shape)
        l = np.expand_dims(b[key],axis=0)

        merged_dict_result[key] = np.concatenate([a[key],l],axis=0)
        # print("the merge_dict[key] shape is:", merge_dict[key].shape)
    return merged_dict_result

class VectorEnvs():
    
    def __init__(self, envs, work_dir, task_name):
        self.envs = envs
        self.work_dir = work_dir
        self.task_name = task_name
        
        from video import VideoRecorder
        self.video_recorder = VideoRecorder(self.work_dir, name=task_name)
        self.video_recorder.init(enabled=True)
        self.ob_dict = None
    
    def is_done(self):
        done = True
        for i,env in enumerate(self.envs):
            done = done and env.all_finished
            # print("the env ",i," is done:", env.all_finished)
        return done
        
    def reset(self, reset_idxs = None, reset_all = False):
        if reset_all == True:
            ob_dict = self.envs[0].reset()
            for key in ob_dict.keys():
                ob_dict[key] = np.expand_dims(ob_dict[key],axis=0)
                
            for env in self.envs[1:]:
                l = env.reset()
                ob_dict = merge_dict(ob_dict,l)
            self.ob_dict = ob_dict
            return ob_dict
        else:
            ob_dict = self.ob_dict
            # print("the ob dict key is:", ob_dict.keys())
            for idx in reset_idxs:
                reset_ob = self.envs[idx].reset()
                for key in ob_dict.keys():
                    ob_dict[key][idx] = reset_ob[key]
            self.ob_dict = ob_dict
            return ob_dict


    def step_without_draw(self, actions, languages):
        ob_dict,reset_flag = self.envs[0].step_without_draw(actions[0], languages[0])
        reset_indexs = []
        if reset_flag:
            reset_indexs.append(0)
            
        for key in ob_dict.keys():
            ob_dict[key] = np.expand_dims(ob_dict[key],axis=0)
            
        for i,env in enumerate(self.envs[1:]):
            next_env_obs, next_env_reset_flag = env.step_without_draw(actions[i+1], languages[i+1])
            ob_dict = merge_dict(ob_dict,next_env_obs)
            if next_env_reset_flag:
                reset_indexs.append(i+1)
        
        self.ob_dict = ob_dict
        return ob_dict, reset_indexs

    def step(self, actions, languages):
        ob_dict,reset_flag = self.envs[0].step(actions[0], languages[0])
        reset_indexs = []
        if reset_flag:
            reset_indexs.append(0)
            
        for key in ob_dict.keys():
            ob_dict[key] = np.expand_dims(ob_dict[key],axis=0)
            
        for i,env in enumerate(self.envs[1:]):
            next_env_obs, next_env_reset_flag = env.step(actions[i+1], languages[i+1])
            ob_dict = merge_dict(ob_dict,next_env_obs)
            if next_env_reset_flag:
                reset_indexs.append(i+1)
        
        self.ob_dict = ob_dict
        return ob_dict, reset_indexs

    def render(self, height=336, width=336, camera_name="agentview"):
        imgs = []
        for env in self.envs:
            imgs.append(env.render(height,width,camera_name))
        return imgs
    
    def close(self):
        total_success_rate = {}
        success_rates_list = []
        idx = 0
        for env in self.envs:
            image_list = env.image_list
            language_list = env.language_set
            
            for img_idx, rollout_imgs in enumerate(image_list):
                for image in rollout_imgs:
                    self.video_recorder.record(image)
                idx += 1
                self.video_recorder.save(f'{self.task_name}_{idx}.mp4')
                self.video_recorder.init(enabled=True)
                
                with open(f"{self.work_dir}/eval_video/{self.task_name}/{self.task_name}_{idx}_language.json", "w") as f:
                    json.dump(language_list[img_idx], f)
                    print("save json")
            env.close()
            mean_success_rate = env.get_success_rate()
            # total_success_rate[env.task_name] = mean_success_rate
            success_rates_list.append(mean_success_rate)
        
        total_success_rate[self.task_name] = np.mean(success_rates_list)
        json_path = os.path.join(self.work_dir, f'{self.task_name}_success_rate.json')
        json.dump(total_success_rate, open(json_path,"w"))
        
