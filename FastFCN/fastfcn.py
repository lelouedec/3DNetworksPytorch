import torch
from torch import nn
import torch.nn.functional as F
import resnet
import numpy as np
from torch.nn.functional import interpolate

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)

class FastFCN(nn.Module):
    def __init__(self,input_size,nb_classes,imsize,backbone,dilated=False, norm_layer=nn.BatchNorm2d,sigmo=True):
        super(FastFCN, self).__init__()


        self.imsize = imsize
        self.sigmo  = sigmo
        if self.sigmo:
            self.sig = nn.Sigmoid()
        ############ ENCODER PART #######################
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=True, dilated=dilated,
                                            norm_layer=norm_layer)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=True, dilated=dilated,
                                            norm_layer=norm_layer)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=True, dilated=dilated,
                                            norm_layer=norm_layer)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        #############JPU part ################
        in_channels = [512, 1024, 2048]
        width=512
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))

        self.dilation1 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))


        ############# Segmentation Head (can be changed)#######################
        self.head = FCNHead(2048, nb_classes, norm_layer)

    def forward(self, x,size):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)



        feats = [self.conv5(c4), self.conv4(c3), self.conv3(c2)]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w),mode='bilinear',align_corners=True)
        feats[-3] = F.interpolate(feats[-3], (h, w),mode='bilinear',align_corners=True)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        c4 = self.head(feat)
        output = interpolate(c4,size,mode='bilinear',align_corners=True)
        if self.sigmo:
            return self.sig(output)
        else:
            return output
        return output
    


if __name__ == "__main__":
    model = FastFCN(3,1,(180,360),'resnet50',sigmo=True)
    input_tensor = torch.from_numpy(np.ones((1,3,180,360)))
    output = model(input_tensor.float(),(360,180))
    print(output.shape)
