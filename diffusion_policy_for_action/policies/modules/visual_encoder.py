import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision import transforms



class Encoder(nn.Module):
    def __init__(self,name, input_channels = 3):
        super().__init__()
        self.net = None
        self.name = name
        self.latent_dim = 512
        self.input_channels = input_channels
    def forward(self, img):
        img_feature = self.net(img)
        
        return img_feature

class ResNet50Encoder(Encoder):
    def __init__(self, name, input_channels = 3,pretrained=False):
        super().__init__(name, input_channels)
        self.name = "resnet50"
        self.hidden_dim = 2048
        
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            net = resnet50(weights=weights, progress=False)
            self.net =  nn.Sequential(*list(net.children())[:-2])
            self.net.eval()
            self.net.requires_grad_(False)
            for param in self.net.parameters():
                param.requires_grad = False
        else:
            net = resnet50(pretrained=False, progress=False)
            self.net = nn.Sequential(*list(net.children())[:-2])

    def forward(self, x):
        x = self.net(x)
        # x = x.flatten(1)
        return x

class ResNet18Encoder(Encoder):
    def __init__(self, name, input_channels=3,pretrained=False):
        super().__init__(name, input_channels)

        self.name = "resnet18"
        self.hidden_dim = 512
        self.pretrained = pretrained
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            net = resnet18(weights=weights, progress=False)
            self.net =  nn.Sequential(*list(net.children())[:-2])
            self.net.eval()
            self.net.requires_grad_(False)
            for param in self.net.parameters():
                param.requires_grad = False
        else:
            net = resnet18(pretrained=False, progress=False)
            self.net = nn.Sequential(*list(net.children())[:-2])

    def forward(self, x):
        if self.pretrained:
            with torch.no_grad():
                x = self.net(x)
        else:
            x = self.net(x)
        # x = x.flatten(1)
        return x
    
from transformers import CLIPProcessor, CLIPModel
class ClipEncoder(Encoder):
    def __init__(self, name, input_channels=3):
        super().__init__(name, input_channels)
        self.name = "clip"
        self.hidden_dim = 512
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        self.model.eval()
        self.model.requires_grad_(False)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, img):
        return self.model.get_image_features(img)

from transformers import AutoImageProcessor, ViTModel
class ViTEncoder(Encoder):
    def __init__(self, name, input_channels=3):
        super().__init__(name, input_channels)
        self.name = "vit"
        self.hidden_dim = 768
        self.net = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        self.net.eval()
        self.net.requires_grad_(False)
        for param in self.net.parameters():
            param.requires_grad = False
    def forward(self, img):
        output = self.net(img)
        # print("output shape is:", output[1])
        # print("output shape is:", output[1].shape)
        return output[0]
        
        
        