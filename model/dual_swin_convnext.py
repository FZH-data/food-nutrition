# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F
from modules.AFPN import AFPN
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from transformers.models.swin.modeling_swin import SwinModel
from model.convnext1 import convnext_small


##############################
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1,output_padding=0, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,output_padding= output_padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.inch = in_planes
    def forward(self, x):

        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FFT(nn.Module):
    def __init__(self,inchannel,outchannel):
        super().__init__()
        self.DWT = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')

        self.IWT = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.conv1 = BasicConv2d(inchannel, inchannel)
        self.conv2 = BasicConv2d(inchannel, inchannel)
        self.conv3 = BasicConv2d(outchannel, outchannel)


    def forward(self, x, y):

        y = self.conv2(y)

        Xl, Xh = self.DWT(x)
        Yl, Yh = self.DWT(y)

        x_y = self.conv1(Xl) + self.conv1(Yl)

        x_m = self.IWT((x_y, Xh))
        x_y=torch.cat([x_m,y],dim=1)
        out = self.conv3(x_y)
        return out


class doubleSwin_convnext_concat(nn.Module):
    def __init__(self,rgb=None,depth=None,clip=None,ingredient=None):
        super(doubleSwin_convnext_concat, self).__init__()

        self.depth = depth
        self.rgb = rgb
        self.clip = clip
        self.ingredient=ingredient

        self.conv1x1 = nn.Conv2d(2048, 1024, kernel_size=1)

        self.con1_1=nn.Conv2d(128,96,kernel_size=1,stride=1,padding=0)
        self.con2_1=nn.Conv2d(256,192,kernel_size=1,stride=1,padding=0)
        self.con3_1=nn.Conv2d(512,384,kernel_size=1,stride=1,padding=0)
        self.con4_1=nn.Conv2d(1024,768,kernel_size=1,stride=1,padding=0)

        self.con1 = nn.Conv2d(256, 192, kernel_size=1, stride=1, padding=0)
        self.con2 = nn.Conv2d(512, 384, kernel_size=1, stride=1, padding=0)
        self.con3 = nn.Conv2d(1024, 768, kernel_size=1, stride=1, padding=0)
        self.con4 = nn.Conv2d(2048, 1536, kernel_size=1, stride=1, padding=0)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))

        self.f1 = nn.Linear(255, 768)
        self.f11 = nn.Linear(255, 768)
        self.f2 = nn.Linear(768, 768)
        self.f22 = nn.Linear(768,768)
        self.f33 = nn.Linear(768,768)


        self.afpn = AFPN([192,384,768,1536],192)

        self.smooth1 = nn.Conv2d(192,192, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)

        self.calorie = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 1))
        self.mass = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 1))
        self.fat = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 1))
        self.carb = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 1))
        self.protein = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 1))

        self.f3 = nn.Linear(1536, 768)

        self.dropout1 = nn.Dropout(p=0.5)

        self.fft1 = FFT(96,192)
        self.fft2 = FFT(192,384)

        self.avgpool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_3 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_4 = nn.AdaptiveAvgPool2d((1, 1))


    def _upsample_add(self, x, y):
        # 将输入x上采样两倍
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear')+y

########################################################
    def forward(self, rgb, depth, clip,ingredient):
                    rgb[0] = self.con1_1(rgb[0])
                    cat1 = self.fft1(rgb[0], depth[0])

                    rgb[1] = self.con2_1(rgb[1])
                    cat2 = self.fft2(rgb[1], depth[1])

                    rgb[2] = self.con3_1(rgb[2])
                    cat3 = torch.cat((rgb[2], depth[2]), dim=1)
                    cat3 = self.con3(clip[2]) + cat3

                    rgb[3] = self.con4_1(rgb[3])
                    cat4 = torch.cat((rgb[3], depth[3]), dim=1)
                    cat4 = self.con4(clip[3]) + cat4

                    list = tuple((cat1, cat2, cat3, cat4))
                    p1, p2, p3, p4 = self.afpn(list)

                    p1 = self.smooth1(p1)
                    p2 = self.smooth2(p2)
                    p3 = self.smooth3(p3)
                    p4 = self.smooth4(p4)

                    p1 = self.avgpool_1(p1)
                    p2 = self.avgpool_2(p2)
                    p3 = self.avgpool_3(p3)
                    p4 = self.avgpool_4(p4)

                    cat_input = torch.stack([p1, p2, p3, p4], axis=1)
                    input = cat_input.view(cat_input.shape[0], -1)

                    input = self.f2(input)

                    ingr = self.f1(ingredient)

                    ingr = self.f22(ingr)
                    ingr1 = self.f33(ingr)
                    input = F.softmax(ingr, dim=1) * input + input
                    input1 = F.softmax(ingr1, dim=1) * input + input

                    output = torch.cat((ingr, input1), dim=1)
                    output = self.f3(output)

                    output = F.relu(output)

                    results = []
                    results.append(self.calorie(output).squeeze())
                    results.append(self.mass(output).squeeze())
                    results.append(self.fat(output).squeeze())
                    results.append(self.carb(output).squeeze())
                    results.append(self.protein(output).squeeze())

                    return results

###########################################################################

# if __name__ == '__main__':
    # rgb = torch.randn([4, 3, 384, 384])
    # depth = torch.randn([4, 3, 384, 384])
    # ingredient=torch.randn([4,255])
    # # model1 = SwinModel.from_pretrained(swin_base_patch4_window12_384_22k.pth)
    # model2 =convnext_small(pretrained=False,in_22k=False)
    # out1 = model1(rgb)
    # r0,r1,r2,r3,r4=out1
    # out2 = model2(depth)
    # d1,d2,d3,d4=out2
    # net_cat = doubleSwin_convnext_concat()
    # model_cat=net_cat([r1,r2,r3,r4],[d1,d2,d3,d4],ingredient)
    # print(model_cat)
