import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import clip

class MotionPrediction(torch.nn.Module):
    def __init__(self, output_size = 20):
        super().__init__()
        vit = models.vit_l_16(pretrained=False,image_size=336)
        self.vit =  nn.Sequential(*list(vit.children())[:-2])
        model, preprocess = clip.load("ViT-B/32")
        self.clip_model = model
    
        self.embed_shape = 1024
    
        self.language_linear = nn.Linear(512, self.embed_shape)
        self.language_vision_attention = nn.MultiheadAttention(embed_dim=self.embed_shape, num_heads=4,dropout=0.3, batch_first=True)
    
        self.output_net = nn.Sequential(
            nn.Linear(self.embed_shape, self.embed_shape // 2),
            nn.ReLU(),
            nn.Linear(self.embed_shape // 2, output_size)
        )
    
    def compute_loss(self, x,  gt):
        output_value = self.forward(x)
        # print(output_value.dtype)
        # print(output_value.shape)
        loss = F.cross_entropy(output_value, gt)
        return loss
    
    def forward(self,x):
        language = x['query']
        with torch.no_grad():
            text_features = self.clip_model.encode_text(language).float()
        vision_feature = self.vit(x['rgb_image'])
        B,D,H,W = vision_feature.shape
        # print("vision", vision_feature.shape)
        language_feature = self.language_linear(text_features)
        language_feature = language_feature.reshape(B,-1,D)
        vision_feature = vision_feature.reshape(B,-1,D)
        # print("vision shape is:", vision_feature.shape)
        attention_output = self.language_vision_attention(query=language_feature,key = vision_feature, value=vision_feature)
        attention_value = attention_output[0]
        
        output_value = self.output_net(attention_value)
        output_value = output_value.squeeze(1)
        return output_value

if __name__ == "__main__":
    model = MotionPrediction().cuda()
    image = torch.randn((2,3,336,336)).to("cuda")
    language = ["move", "put the bag"]
    language_token = clip.tokenize(language).to("cuda")
    print(language_token.shape)
    inputs = {
        "query": language_token,
        "rgb_image": image
    }
    output = model(inputs)
    labels = torch.randint(0,20, (2,)).cuda()
    loss = torch.nn.functional.cross_entropy(output, labels)
    print(loss)
    