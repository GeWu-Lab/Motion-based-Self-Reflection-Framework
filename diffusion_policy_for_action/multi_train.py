import warnings
import os
local_rank = int(os.environ["LOCAL_RANK"])
import hydra
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import policies.transformer_policy
import policies.transformer_policy_codebook
import time
import utils
import trajectory_data
import wandb
from omegaconf import OmegaConf
from easydict import EasyDict
import yaml

from torch.utils.data.distributed import DistributedSampler
import random
from tqdm import tqdm

class WorkSpace:
    def __init__(self, cfg):
        self.work_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.cfg = cfg
        # self.device = cfg.device

        torch.cuda.set_device(local_rank)
        self.device = torch.device('cuda', local_rank)
        print("the local rank is:", local_rank)
        
        
        self.loss_threshold = cfg.loss_threshold

        # dataset = hydra.utils.instantiate(cfg.dataset)
        train_dataset = eval(cfg.dataset._target_)(**cfg.dataset.kwargs)
        
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        self.train_dataloader = train_dataset.get_dataloader(num_workers= cfg.num_workers ,sampler = train_sampler)
        # self.eval_dataloader = eval_dataset.get_dataloader()
        self.train_sampler = train_sampler
        
        shape_meta = train_dataset.get_shape_meta()
        print("the shape meta is:", shape_meta)
        self.policy = eval(cfg.policy._target_)(cfg.policy, shape_meta).to(self.device)
        # self.policy =  self.policy.cuda()

        self.policy = torch.nn.parallel.DistributedDataParallel(self.policy, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        self.augmentation = trajectory_data.construct_augmentation(cfg.dataset.kwargs.augmentation_cfg, self.cfg.policy.encoder.network_kwargs.name)
        self.optimizer = eval(cfg.optimizer._target_)(self.policy.parameters(), **cfg.optimizer.network_kwargs)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, cfg.epoch)
        
        # self.policy = hydra.utils.instantiate(cfg.policy).to(self.device)
        if local_rank == 0:

            self.sw = SummaryWriter(os.path.join(self.work_dir, 'tb'))
        else:
            self.sw = None
        yaml_config = OmegaConf.to_yaml(self.cfg)



    def save_snapshot(self, idx):
        if torch.distributed.get_rank() == 0:
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
        
        self.policy.load_state_dict(data['policy'])

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
        # self.policy = torch.nn.DataParallel(self.policy).to(self.cfg.device)
        
        for epoch in range(self.cfg.epoch):
            self.train_sampler.set_epoch(epoch)
            train_losses = []
            a = time.time()
            for batch in tqdm(self.train_dataloader):
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
                
                gt_action = batch['actions'].to(self.cfg.device)
                # print("the to cuda time is:", time.time() - start_time)
                l = time.time()
                self.optimizer.zero_grad()
                # print("gt action is:", gt_action.shape)
                # print("gt action is:", gt_action.dtype)
                loss = self.policy.module.compute_loss(obs, gt_action)

                # dist = self.policy.forward(obs)
                # loss = self.policy.policy_head.loss_fn(dist, gt_action, reduction = "mean")
            
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                end_time = time.time()
                duration = end_time - start_time
                # print("the backward time is:", time.time() - l)
                # print("the train time is:", duration)
                print("the loss is:", loss)
                if self.sw is not None:
                    self.sw.add_scalar('Loss/train', loss, train_step)
                # self.run.log({"Loss/train": loss, "Duration/train": duration}, step=train_step)
            epoch_loss = np.mean(train_losses)
            
            if epoch_loss < best_train_loss:
                best_train_loss = epoch_loss
                self.save_snapshot("best_train")

            if train_first_flag and epoch_loss < self.loss_threshold:
                train_first_flag = False
                self.save_snapshot("first_train_threshold")
                
            if epoch % 50 == 0:
                print("i save the snapshot")
                self.save_snapshot(epoch)
                # eval_loss = self.eval()
                # if eval_loss < best_eval_loss:
                #     best_eval_loss = eval_loss
                #     self.save_snapshot("best")


        if self.sw is not None:
            self.sw.close()
        
        # self.run.finish()



@hydra.main(config_path='cfgs', config_name='multi_config',version_base=None)
def main(cfg):
    torch.distributed.init_process_group(backend='nccl')
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    w = WorkSpace(cfg)
    w.train()



if __name__ == '__main__':
	main()