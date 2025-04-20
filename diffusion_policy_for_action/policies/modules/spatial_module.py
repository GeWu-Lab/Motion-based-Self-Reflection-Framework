from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from torch import einsum
# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class SpatialSoftmax(nn.Module):
    """
    The spatial softmax layer (https://rll.berkeley.edu/dsae/dsae.pdf)
    """

    def __init__(self, in_c, in_h, in_w, num_kp=None):
        super().__init__()
        self._spatial_conv = nn.Conv2d(in_c, num_kp, kernel_size=1)

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, in_w).float(),
            torch.linspace(-1, 1, in_h).float(),
        )

        pos_x = pos_x.reshape(1, in_w * in_h)
        pos_y = pos_y.reshape(1, in_w * in_h)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

        if num_kp is None:
            self._num_kp = in_c
        else:
            self._num_kp = num_kp

        self._in_c = in_c
        self._in_w = in_w
        self._in_h = in_h

    def forward(self, x):
        # print("the x shape is:", x.shape)
        assert x.shape[1] == self._in_c
        assert x.shape[2] == self._in_h
        assert x.shape[3] == self._in_w

        h = x
        if self._num_kp != self._in_c:
            h = self._spatial_conv(h)
        h = h.contiguous().view(-1, self._in_h * self._in_w)

        attention = F.softmax(h, dim=-1)
        keypoint_x = (
            (self.pos_x * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        )
        keypoint_y = (
            (self.pos_y * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        )
        keypoints = torch.cat([keypoint_x, keypoint_y], dim=1)
        return keypoints

class SpatialProjection(nn.Module):
    def __init__(self, input_shape, out_dim):
        super().__init__()

        assert (
            len(input_shape) == 3
        ), "[error] spatial projection: input shape is not a 3-tuple"
        in_c, in_h, in_w = input_shape
        num_kp = out_dim // 2
        self.out_dim = out_dim
        self.spatial_softmax = SpatialSoftmax(in_c, in_h, in_w, num_kp=num_kp)
        self.projection = nn.Linear(num_kp * 2, out_dim)

    def forward(self, x):
        out = self.spatial_softmax(x)
        out = self.projection(out)
        return out

    def output_shape(self, input_shape):
        return input_shape[:-3] + (self.out_dim,)



class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class Perceiver(nn.Module):
    def __init__(self, query_token = 128, hidden_dim = 512, num_heads = 4, num_layers=1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.model = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.query = nn.Embedding(query_token, hidden_dim)
        
    
    def forward(self, x):
        B, L, D = x.shape
        query = self.query.weight.repeat(B, 1, 1)
        out = self.model(query, x)
        
        out = torch.mean(out, dim=1) # B, D
        
        return out

if __name__ == "__main__":
    weights = ResNet50_Weights.DEFAULT
    net = resnet50(weights=weights, progress=False)
    net =  nn.Sequential(*list(net.children())[:-2])
    
    print(net)
    
    x = torch.randn(1,3,224,224)
    
    y = net(x)
    
    print(y.shape)
    projection = SpatialProjection(input_shape=y.shape[1:], out_dim=512)
    
    out = projection(y)
    
    print(out.shape)