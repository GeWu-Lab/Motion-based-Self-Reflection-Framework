from policies.policy import BasePolicy
from policies.modules.extra_encoder import StateEncoder

import torch.nn as nn
import torch
import hydra
from policies.modules.visual_encoder import *
import policies
from policies.modules.transformer_modules import *
import policies.policy_head
import utils
import time
from policies.policy_head import GMMHead, DeterministicHead, QueryDecoderDeterministicHead, QueryDecoderGMMHead
from policies.modules.spatial_module import SpatialProjection, Perceiver
from policies.diffusion_policy import ConditionalUnet1D

class BCTransformerPolicy(BasePolicy):
    def __init__(self, cfg, shape_meta):
        super().__init__()
        self.extra_dim = cfg.extra_dim
        self.cfg = cfg
        # print("cfg is:", cfg)
        # add vision encoder
        self.image_encoders = {}
        self.image_spatial_encoders = {}
        n_modal = 0 # init language
        
        
        self.use_language_idx = cfg.use_language_idx
        self.language_codebook = nn.Embedding(cfg.language_codebook_size, cfg.hidden_dim)
        
        for name in shape_meta.keys():
            # print("cfg is:", cfg)
            encoder_cfg = cfg.encoder
            if "depth" in name:
                encoder_cfg.network_kwargs.input_channels = 1
            self.image_encoders[name] = eval(encoder_cfg._target_)(**encoder_cfg.network_kwargs).to(cfg.device)
            
            if self.cfg.use_flatten:
                self.image_spatial_encoders[name] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(cfg.visual_hidden_dim * 3 * 3, cfg.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
                ).to(cfg.device)

            elif "resnet" in self.image_encoders[name].name or "clip" in self.image_encoders[name].name:
                self.image_spatial_encoders[name] = SpatialProjection((cfg.visual_hidden_dim, 3, 3), out_dim=cfg.hidden_dim).to(cfg.device)
            else:
                # TODO: Transformer Projection
                self.image_spatial_encoders[name] = Perceiver(query_token=64, hidden_dim=cfg.hidden_dim, num_heads=4, num_layers=1).to(cfg.device)
            n_modal += 1
            
        encoder_list = []
        spatial_encoder_list = []
        for name in shape_meta.keys():
            encoder_list.append((name, self.image_encoders[name]))
            spatial_encoder_list.append((name, self.image_spatial_encoders[name]))
            
        self.encoders = nn.ModuleDict(encoder_list)
        self.spatial_encoders = nn.ModuleDict(spatial_encoder_list)
        # add the state encoder
        if self.extra_dim != 0:
            self.extra_encoder = StateEncoder(cfg.num_layers, cfg.hidden_dim, self.extra_dim)
            n_modal += 1
        else:
            self.extra_encoder = None
        
        # add position embedding
        cfg.position_encoding.network_kwargs.input_size = cfg.hidden_dim
        self.temporal_position_encoding_fn = eval(
            cfg.position_encoding._target_
        )(**cfg.position_encoding.network_kwargs)

        # add temporal transformer        
        self.temporal_transformer = TransformerDecoder(
            input_size=cfg.hidden_dim,
            num_layers=cfg.transformer_num_layers,
            num_heads=cfg.transformer_num_heads,
            head_output_size=cfg.transformer_head_output_size,
            mlp_hidden_size=cfg.transformer_mlp_hidden_size,
            dropout=cfg.transformer_dropout,
        )

        # self.language_linear = nn.Sequential(
        #     nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(self.cfg.hidden_dim // 2, self.cfg.hidden_dim),
        #     nn.ReLU()
        # )

        # convert action chunking
        # self.convert = nn.MultiheadAttention(cfg.hidden_dim , num_heads=2, batch_first = True)
        # self.query_embed = nn.Embedding(cfg.action_length, cfg.hidden_dim)

        # if cfg.policy_head.network_kwargs.get("global_cond_dim") is not None:
        #     self.cfg.policy_head.network_kwargs.global_cond_dim = cfg.hidden_dim

        self.n_modal = n_modal

        # add policy head
        self.policy_head = eval(self.cfg.policy_head._target_)(
            **self.cfg.policy_head.network_kwargs
        )
        # print("the policy head is:", self.policy_head)
        
        if isinstance(self.policy_head, QueryDecoderDeterministicHead) or isinstance(self.policy_head, QueryDecoderGMMHead):
            self.is_query_decoder = True
        else:
            self.is_query_decoder = False
        
        self.latent_queue = []
        # for i in range(cfg.frame_length):
        #     self.latent_queue.append(torch.zeros(1,1,n_modal,cfg.hidden_dim).to(cfg.device))

        self.language_queue = []
        
    def reset(self):
        self.latent_queue = []
        # for i in range(self.cfg.frame_length):
        #     self.latent_queue.append(torch.zeros(1,1,self.n_modal,self.cfg.hidden_dim).to(self.cfg.device))

        self.language_queue = []

    def spatial_encoder(self, obs):

        encoded_features = []
        
        # 1. encode the visual observation
        for img_name in self.encoders.keys():
            x = obs[img_name]
            B,T,C,H,W = x.shape
            # print("the shape is:", x.shape)
            # print("the name is:", img_name)
            # print("the x device is:", x.device)
            img_encoded = self.encoders[img_name](x.view(B*T,C,H,W))
            # print("the img_encoded shape is:", img_encoded.shape)
            
            img_encoded = self.spatial_encoders[img_name](img_encoded).view(B,T,1,-1)
            # .view(B,T,1,-1)
            # print("the img_encoded shape is:", img_encoded.shape)
            encoded_features.append(img_encoded)


        B,T,C,H,W = obs['third_rgb'].shape
        # 2. encode the state observation
        if self.extra_dim != 0:
            state_encoded = self.extra_encoder(obs['states']).view(B,T,1,-1)
            # print("the state_encoded shape is:", state_encoded.shape)
            encoded_features.append(state_encoded)
        
        # print("the language feature shape is:", obs['language_feature'].shape)
        # print("encoded_features shape is:", encoded_features[0].shape)
        # language_feature = obs['language_feature'].view(B,T,1,-1)
        # encoded_features.append(language_feature)
        
        
        concat_feature = torch.cat(encoded_features, dim=-2) # (B, T, num_modalities, E)
        
        return concat_feature
    
    def temporal_encoder(self, x):
        B,T,N,E = x.shape
        pos_emb = self.temporal_position_encoding_fn(x)

        sh = x.shape
        x = x + pos_emb.unsqueeze(1)  # (B, T, num_modality, E)
        self.temporal_transformer.compute_mask(x.shape)
        # print("the mask is:", self.temporal_transformer.mask)
        x = utils.join_dimensions(x, 1, 2) # N, T * num_modality, E
        
        # print("the joint x shape is:", x.shape)
        x = self.temporal_transformer(x)
        
        x = x.view(*sh)  # N,T,num_modality,E
        if self.is_query_decoder:
            return x
        else:
            return x[:,:,0]

    def add_obs_history(self,obs):
        with torch.no_grad():
            x = self.spatial_encoder(obs)
            self.latent_queue.append(x)
            while len(self.latent_queue) < self.cfg.frame_length:
                self.latent_queue.append(x)
            if len(self.latent_queue) > self.cfg.frame_length:
                self.latent_queue.pop(0)
        

    def get_action(self, obs, reset_indexs = []):
        self.eval()
        with torch.no_grad():
            x = self.spatial_encoder(obs)
            # print("the x shape is:", x.shape)
            B = x.shape[0]
            
            self.latent_queue.append(x)
            while len(self.latent_queue) < self.cfg.frame_length:
                self.latent_queue.append(x)
            
            if self.use_language_idx:
                language_feature = self.language_codebook(obs['language_idx'])
            else:
                language_feature = obs['language_feature']
            # print("the language feature shape is:", language_feature.shape)
            
            self.language_queue.append(language_feature.view(B,1,-1))
            while len(self.language_queue) < self.cfg.frame_length:
                self.language_queue.append(language_feature.view(B,1,-1))
            
            if len(self.language_queue) > self.cfg.frame_length:
                self.language_queue.pop(0)
            
            if len(self.latent_queue) > self.cfg.frame_length:
                self.latent_queue.pop(0)
            
            # reset of multithread
            for reset_idx in reset_indexs:
                for l in range(len(self.latent_queue)):
                    self.latent_queue[l][reset_idx] = x[reset_idx]
                    self.language_queue[l][reset_idx] = language_feature[reset_idx].view(1,1,-1)
            
            x = torch.cat(self.latent_queue, dim=1)
            language_features = torch.cat(self.language_queue, dim=1)
            # print("the x shape is:", x.shape)
            x = self.temporal_encoder(x)
            B,T,E = x.shape

            global_feature = {
                "obs_feature": x,
                "language_feature": language_features
            }
            
            x = global_feature
            # x = torch.cat([x, language_features], dim=-1)
            
            # language_feature = obs['language_feature'].view(B,-1)
            # concat_feature = torch.cat([x, language_feature], dim=-1)
            # x is B,T,E
            dist = self.policy_head.get_action(x)
            # print("the dist shape is:", dist.shape)
            if isinstance(self.policy_head, DeterministicHead):
                B,T,L = x.shape
                dist = dist.view(B,T,self.cfg.action_length,-1)
                dist = dist[:,-1]
            if isinstance(self.policy_head, GMMHead):
                dist = dist[:,-1]
        return dist
            
    def forward(self, obs):
        a = time.time()
        # obs = self.map_tensor_to_device(obs)
        #
        x = self.spatial_encoder(obs)
        b = time.time()
        x = self.temporal_encoder(x)
        B,T,E = x.shape
        c = time.time()
        
        # language_features = obs['language_feature'].view(B,T,-1)
        if self.use_language_idx:
        
            language_idx = obs['language_idx']
            language_feature = self.language_codebook(language_idx)
        else:
            language_feature = obs['language_feature']
        # x = torch.cat([x, language_feature], dim=-1)

        global_feature = {
            "obs_feature": x,
            "language_feature": language_feature
        }

        x = global_feature
        # x is (B,T,E)
        if isinstance(self.policy_head, ConditionalUnet1D):
            return x
        else:
            dist = self.policy_head(x)
            if isinstance(self.policy_head, DeterministicHead):
                B,T,L = x.shape
                # print("the dist shape is:", dist.shape)
                dist = dist.view(B,T,self.cfg.action_length,-1)
            
            return dist

    def compute_loss(self, obs, gt_action, reduction="mean"):
        dist = self.forward(obs)
        # print("the gt_action is:", gt_action.shape)
        # print("the predicted action is:", dist.shape)
        l = time.time()
        loss = self.policy_head.loss_fn(dist, gt_action, reduction)
        # print("the diffusion time is:", time.time() - l) 
        return loss
    