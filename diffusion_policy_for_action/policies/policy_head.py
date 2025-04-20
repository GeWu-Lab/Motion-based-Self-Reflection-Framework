
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import utils
from policies.modules.transformer_modules import SinusoidalPositionEncoding



class DeterministicHead(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024, num_layers=2):

        super().__init__()
        self.name = "DeterministicHead"
        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]
        
        layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        # print("the y shape is:", y.shape)
        return y

    def get_action(self, x):
        action = self.forward(x)
        return action

    def loss_fn(self, action, gt_action, reduction="mean"):
        # print("the gt_action shape is:", gt_action.shape)
        # print("the action shape is:", action.shape)
        loss = F.mse_loss(action, gt_action, reduction=reduction)
        return loss

import torch
import torch.nn as nn

class QueryDecoderDeterministicHead(nn.Module):
    def __init__(self, input_size, output_size,action_length, num_layers, num_heads, hidden_size=128):
        super().__init__()
        self.projector = nn.Linear(input_size, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        model = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # self.query_embed = nn.Embedding(action_length, input_size)

        position_emd_creator  = SinusoidalPositionEncoding(input_size = hidden_size)
        self.query_embed = position_emd_creator(torch.zeros(1, action_length)).unsqueeze(0).cuda()
        
        
        self.action_length = action_length
        self.output_size = output_size
        
        self.model = model
        self.name = "QueryDecoderDeterministicHead"
        
        self.policy_head = DeterministicHead(hidden_size, output_size, hidden_size, num_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N_MODAL, L)

        x = self.projector(x)
        B, T, N, L = x.shape
        # print("the query shape is:", self.query_embed.shape)
        query = self.query_embed.repeat(B * T, 1, 1)

        x = x.view(B * T, N, L)
        # print("the x shape is:", x.shape)
        # print("the query shape is:", query.shape)
        out = self.model(query, x)
        # print("the out shape is:", out.shape)
        
        out = self.policy_head(out)

        out = out.view(B, T, self.action_length, self.output_size)

        # print("the out shape is:", out.shape)
        return out

    def get_action(self, x):
        action = self.forward(x)
        return action

    def loss_fn(self, action, gt_action, reduction="mean"):
        # print("the gt_action shape is:", gt_action.shape)
        # print("the action shape is:", action.shape)
        loss = F.mse_loss(action, gt_action, reduction=reduction)
        return loss

class QueryDecoderGMMHead(nn.Module):
    def __init__(self, input_size, output_size,action_length, num_layers, num_heads, hidden_size=128, loss_coef=1):
        super().__init__()
        
        self.projector = nn.Linear(input_size, hidden_size)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        model = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        position_emd_creator  = SinusoidalPositionEncoding(input_size = hidden_size)
        self.query_embed = position_emd_creator(torch.zeros(1, action_length)).unsqueeze(0).cuda()
        # self.query_embed = nn.Embedding(action_length, input_size)
        
        self.action_length = action_length
        self.loss_coef = loss_coef
        
        self.model = model
        self.name = "QueryDecoderGMMHead"
        
        self.policy_head = GMMHead(hidden_size, output_size, hidden_size, num_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N_MODAL, L)
        x = self.projector(x)
        B, T, N, L = x.shape
        query = self.query_embed.repeat(B * T, 1, 1)

        x = x.view(B * T, N, L)
        out = self.model(query, x)
        # output: (B * T, action_length, L)
        
        out = self.policy_head(out)

        # print("the out shape is:", out.shape)
        return out
    
    def get_action(self, x):
        action = self.forward(x)
        return action

    def loss_fn(self, gmm, target, reduction="mean"):
        # print("the target is:", target.shape)
        # print("the gmm is:", gmm)
        B,T,A,L = target.shape
        target = target.view(B*T,A,L)
        # target = target.reshape(128,5,70)
        log_probs = gmm.log_prob(target)
        # print("the log probs is:", log_probs.shape)
        loss = -log_probs
        if reduction == "mean":
            return loss.mean() * self.loss_coef
        elif reduction == "none":
            return loss * self.loss_coef
        elif reduction == "sum":
            return loss.sum() * self.loss_coef
        else:
            raise NotImplementedError



class GMMHead(nn.Module):
    def __init__(
        self,
        # network_kwargs
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        min_std=0.0001,
        num_modes=5,
        activation="softplus",
        low_eval_noise=False,
        # loss_kwargs
        loss_coef=1.0,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.output_size = output_size
        self.min_std = min_std
        self.name = "GMMHead"
        
        if num_layers > 0:
            sizes = [input_size] + [hidden_size] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        self.mean_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logstd_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logits_layer = nn.Linear(hidden_size, num_modes)

        self.low_eval_noise = low_eval_noise
        self.loss_coef = loss_coef

        if activation == "softplus":
            self.actv = F.softplus
        else:
            self.actv = torch.exp
    
    def get_action(self, x):
        action = self.forward(x)
        action = action.sample()
        return action
    
    def forward_fn(self, x):
        # x: (B, input_size)
        share = self.share(x)
        means = self.mean_layer(share).view(-1, self.num_modes, self.output_size)
        
        means = torch.tanh(means)
        
        logits = self.logits_layer(share)

        if self.training or not self.low_eval_noise:
            logstds = self.logstd_layer(share).view(
                -1, self.num_modes, self.output_size
            )
            stds = self.actv(logstds) + self.min_std
        else:
            stds = torch.ones_like(means) * 1e-4
        return means, stds, logits

    def forward(self, x):
        # print("the x shape is:", x.shape)
        if x.ndim == 3:
            means, scales, logits = utils.time_distributed(x, self.forward_fn)
        elif x.ndim < 3:
            means, scales, logits = self.forward_fn(x)

        
        compo = D.Normal(loc=means, scale=scales)
        compo = D.Independent(compo, 1)
        mix = D.Categorical(logits=logits)
        gmm = D.MixtureSameFamily(
            mixture_distribution=mix, component_distribution=compo
        )
        return gmm

    def loss_fn(self, gmm, target, reduction="mean"):
        # print("the target is:", target.shape)

        # target = target.reshape(128,5,70)
        log_probs = gmm.log_prob(target)
        # print("the log probs is:", log_probs.shape)
        loss = -log_probs
        if reduction == "mean":
            return loss.mean() * self.loss_coef
        elif reduction == "none":
            return loss * self.loss_coef
        elif reduction == "sum":
            return loss.sum() * self.loss_coef
        else:
            raise NotImplementedError

