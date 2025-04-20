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
from policies.policy_head import GMMHead, DeterministicHead, QueryDecoderDeterministicHead, QueryDecoderGMMHead
from policies.modules.spatial_module import SpatialProjection, Perceiver

class RNNPolicy(BasePolicy):
    def __init__(self, cfg, shape_meta):
        super().__init__()
        self.extra_dim = cfg.extra_dim
        self.cfg = cfg
        # print("cfg is:", cfg)
        # add vision encoder
        self.device = cfg.device
        rnn_input_size = 0
        self.image_encoders = {}
        self.image_spatial_encoders = {}

        for name in shape_meta.keys():
            # print("cfg is:", cfg)
            encoder_cfg = cfg.encoder
            if "depth" in name:
                encoder_cfg.network_kwargs.input_channels = 1
            self.image_encoders[name] = eval(encoder_cfg._target_)(**encoder_cfg.network_kwargs).to(cfg.device)
            
            rnn_input_size += cfg.hidden_dim
            
            if "resnet" in self.image_encoders[name].name:
                # print("the shape meta is:", shape_meta[name])
                
                self.image_spatial_encoders[name] = SpatialProjection((cfg.hidden_dim, 7, 7), out_dim=cfg.hidden_dim).to(cfg.device)
            else:
                # TODO: Transformer Projection
                self.image_spatial_encoders[name] = Perceiver(query_token=128, hidden_dim=cfg.hidden_dim, num_heads=4, num_layers=1).to(cfg.device)
                
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
            rnn_input_size += cfg.hidden_dim
        else:
            self.extra_encoder = None

    
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=cfg.rnn_hidden_size,
            num_layers=cfg.rnn_num_layers,
            batch_first=True,
            dropout=cfg.rnn_dropout,
            bidirectional=cfg.rnn_bidirectional,
        )

        ### 4. use policy head to output action
        self.D = 2 if cfg.rnn_bidirectional else 1
        policy_head_kwargs = cfg.policy_head.network_kwargs
        policy_head_kwargs.input_size = self.D * cfg.rnn_hidden_size
        
        
        self.policy_head = eval(self.cfg.policy_head._target_)(
            **self.cfg.policy_head.network_kwargs
        )

        if isinstance(self.policy_head, QueryDecoderDeterministicHead) or isinstance(self.policy_head, QueryDecoderGMMHead):
            self.is_query_decoder = True
        else:
            self.is_query_decoder = False

        self.eval_h0 = None
        self.eval_c0 = None
        
    def spatial_encoder(self, obs):

        encoded_features = []
        
        # 1. encode the visual observation
        for img_name in self.encoders.keys():
            x = obs[img_name]
            B,T,C,H,W = x.shape
            # print("the x device is:", x.device)
            img_encoded = self.encoders[img_name](x.reshape(B*T,C,H,W))
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
        
        concat_feature = torch.cat(encoded_features, dim=-2) # (B, T, num_modalities, E)
        
        return concat_feature
        
    def forward(self, obs):
        
        x = self.spatial_encoder(obs) # B,T,N_modality,D
        B,T,N,D = x.shape
        encoded = x.view(B,T,N * D)
        
        if self.training:
            h0 = torch.zeros(
                self.D * self.cfg.rnn_num_layers,
                encoded.shape[0],
                self.cfg.rnn_hidden_size,
            ).to(self.device)
            c0 = torch.zeros(
                self.D * self.cfg.rnn_num_layers,
                encoded.shape[0],
                self.cfg.rnn_hidden_size,
            ).to(self.device)
            output, (hn, cn) = self.rnn(encoded, (h0, c0))
            print("the output shape is:", output.shape)
            print("the hn shape is:", hn.shape)

        else:
            if self.eval_h0 is None:
                self.eval_h0 = torch.zeros(
                    self.D * self.cfg.policy.rnn_num_layers,
                    encoded.shape[0],
                    self.cfg.rnn_hidden_size,
                ).to(self.device)
                self.eval_c0 = torch.zeros(
                    self.D * self.cfg.policy.rnn_num_layers,
                    encoded.shape[0],
                    self.cfg.rnn_hidden_size,
                ).to(self.device)
            output, (h1, c1) = self.rnn(encoded, (self.eval_h0, self.eval_c0))
            self.eval_h0 = h1.detach()
            self.eval_c0 = c1.detach()
        
        dist = self.policy_head(output)
        if isinstance(self.policy_head, DeterministicHead):
            dist = dist.view(B,T,self.cfg.action_length,-1)
            
        return dist
    
    def reset(self):
        self.eval_h0 = None
        self.eval_c0 = None

    def get_action(self, data):
        self.eval()
        
        with torch.no_grad():
            dist = self.forward(data)
        
        return dist

    def compute_loss(self, obs, gt_action, reduction="mean"):
        dist = self.forward(obs)
        # gt_action = self.map_tensor_to_device(gt_action)
        # print("the gt_action shape is:", gt_action.shape)
        # print("the dist shape is:", dist.shape)
        loss = self.policy_head.loss_fn(dist, gt_action, reduction)
        # print("the loss is:", loss)
        return loss