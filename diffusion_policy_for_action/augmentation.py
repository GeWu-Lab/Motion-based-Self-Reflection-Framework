import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)



class ImgColorJitterAug(torch.nn.Module):
    """
    Conduct color jittering augmentation outside of proposal boxes
    """

    def __init__(
        self,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.05,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon

    def forward(self, x):
        if self.training and np.random.rand() > self.epsilon:
            out = self.color_jitter(x)
        else:
            out = x
        return out

    def output_shape(self, input_shape):
        return input_shape