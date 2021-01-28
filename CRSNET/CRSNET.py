import torch
from torch import nn
import torch.nn.functional as F
import  numpy as np
import torchvision.models as models


class CRSNET(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(CRSNET, self).__init__()
        self.base = models.vgg16().features
        self.encoder = []
        for i in range(0,23):
            self.encoder.append(self.base[i])
        
        self.encoder = nn.Sequential(*self.encoder)
        del self.base

        self.decoder = nn.Sequential(
            nn.Conv2d(512,512,3,dilation=2,padding=2), 
            nn.ReLU(),
            nn.Conv2d(512,512,3,dilation=2,padding=2), 
            nn.ReLU(),
            nn.Conv2d(512,256,3,dilation=2,padding=2), 
            nn.ReLU(),
            nn.Conv2d(256,128,3,dilation=2,padding=2), 
            nn.ReLU(),
            nn.Conv2d(128,64,3,dilation=2,padding=2), 
            nn.ReLU(),
            nn.Conv2d(64,64,3,dilation=2,padding=2), 
            nn.ReLU(),
        )
       
        self.out_conv = nn.Conv2d(64,n_classes,1)
    def forward(self,x):
        x = self.encoder(x)
        print(x.shape)
        x = self.decoder(x)
        return self.out_conv(x)



if __name__ == '__main__':
    model = CRSNET(3,1)
    print(model)
    input_tensor = torch.ones((1,3,224,224))
    output = model(input_tensor)
    print(output.shape)