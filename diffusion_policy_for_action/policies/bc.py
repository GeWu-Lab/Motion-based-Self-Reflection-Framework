from policies.policy import BasePolicy
from policies.modules.visual_encoder import *
import hydra
from policies.policy_head import *
from itertools import chain

import torch.nn.functional as F
import utils
import torch.nn as nn
import torch

class BCStatePolicy(BasePolicy):
    def __init__(self, hidden_dim, action_shape, lr, policy_head, num_layers, extra_dim=30):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.action_shape = action_shape
        self.policy_head = policy_head
        self.num_layers = num_layers
        self.extra_dim = extra_dim
        

        layers = [nn.Linear(self.extra_dim, self.hidden_dim), nn.ReLU()]
        if num_layers > 0:
            for _ in range(num_layers - 1):
                layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()])
        
        self.extra_encoder = nn.Sequential(*layers)

        self.actor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Tanh(),
            self.policy_head
        )

        self.optimizer = torch.optim.Adam(
            chain(self.extra_encoder.parameters(),self.actor.parameters()), lr=lr
        )

        self.apply(utils.weight_init)

    def get_action(self, obs):

        state_obs = self.extra_encoder(state_obs)
        action_obs = state_obs

        action = self.actor(action_obs)
        return action
    
    def update_actor(self, obs, gt_action):
        gt_action = gt_action.to("cuda")

        state_obs = obs['state'].to("cuda")
        state_obs = self.extra_encoder(state_obs)
        action_obs = state_obs

            
        action = self.actor(action_obs)
        loss = self.policy_head.loss_fn(action, gt_action)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    



class BCPolicy(BasePolicy):
    def __init__(self, hidden_dim, action_shape, lr, encoder, policy_head, extra_dim=0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.action_shape = action_shape
        self.image_encoder = encoder
        self.policy_head = policy_head
        
        self.extra_dim = extra_dim
        

        if self.extra_dim != 0:
            self.extra_encoder = nn.Sequential(
                nn.Linear(self.extra_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.image_encoder.hidden_dim)
            )
        

        self.actor = nn.Sequential(
            nn.Linear(self.image_encoder.hidden_dim if extra_dim == 0 else self.image_encoder.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Tanh(),
            self.policy_head
        )


        if self.extra_dim != 0:
            self.optimizer = torch.optim.Adam(
                chain(self.image_encoder.parameters(),self.extra_encoder.parameters(),self.actor.parameters()), lr=lr
            )
        else:
            self.optimizer = torch.optim.Adam(
            chain(self.image_encoder.parameters(),self.actor.parameters()), lr=lr
        )
        

    def get_action(self, obs):

        visual_obs = obs['visual']
        visual_obs = self.image_encoder(visual_obs)

        if self.extra_dim != 0:
            state_obs = obs['state']

            state_obs = self.extra_encoder(state_obs)
            
            action_obs = torch.cat([visual_obs, state_obs], dim=-1)
        else:
            action_obs = visual_obs


        action = self.actor(action_obs)
        if self.policy_head.name == "GMMHead":
            action = action.mean
            
        return action
    
    def update_actor(self, obs, gt_action):
        gt_action = gt_action.to("cuda")
        visual_obs = obs['visual'].to("cuda")
        
        visual_obs = self.image_encoder(visual_obs)
        
        if self.extra_dim != 0:
            state_obs = obs['state'].to("cuda")
            state_obs = self.extra_encoder(state_obs)
            action_obs = torch.cat([visual_obs, state_obs], dim=-1)
        else:
            action_obs = visual_obs
            
        action = self.actor(action_obs)
        loss = self.policy_head.loss_fn(action, gt_action)
        # print("the loss is:", loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

