import torch
import torch.nn.functional as F
from torch.optim import Adam
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn
from torchvision import models
from COA_RGBD_SOD.Shunted_Transformer_master.SSA import *
from COA_RGBD_SOD.Backbone.p2t import *
from thop import profile
from COA_RGBD_SOD.al.models.mix_transformer import *
from COA_RGBD_SOD.al.models.Second_model.module_S import Decoder, block, block4

class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)
        self.point_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0, groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)

        return out


class CoNet(nn.Module):

    def __init__(self):
        super(CoNet, self).__init__()
        self.conv_RGB = p2t_tiny()
        self.conv_depth = p2t_tiny()
        self.channels = [48, 96, 240, 384]
        self.fusion4 = block4(self.channels[3], self.channels[3])
        self.fusion3 = block(self.channels[2], self.channels[3])
        self.fusion2 = block(self.channels[1], self.channels[2])
        self.fusion1 = block(self.channels[0], self.channels[1])
        self.decoder1 = Decoder()
        self.decoder2 = Decoder()
        self.conv64 = DepthWiseConv(self.channels[0], 1)
        self.conv64_1 = DepthWiseConv(self.channels[0], 1)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x, depth):

        e1_rgb, e2_rgb, e3_rgb, e4_rgb = self.conv_RGB(x)
        e1_depth, e2_depth, e3_depth, e4_depth = self.conv_depth(depth)
        f4, a4, fr4, fd4 = self.fusion4(e4_rgb, e4_depth)
        f3, a3, fr3, fd3 = self.fusion3(e3_rgb, e3_depth, f4)
        f2, a2, fr2, fd2 = self.fusion2(e2_rgb, e2_depth, f3)
        f1, a1, fr1, fd1 = self.fusion1(e1_rgb, e1_depth, f2)

        fusion1, fusion2, fusion3 = self.decoder1(f4, f3, f2, f1)
        S = self.decoder2(f4, fusion3, fusion2, fusion1, fusion1)
        S = self.conv64(self.up4(S))
        S1 = self.conv64_1(self.up4(fusion1))

        return S, S1, f1, f2, f3, f4, a1, a2, a3, a4, fusion1, fusion2, fusion3


    def load_pre(self, pre_model):
        save_model = torch.load(pre_model)
        model_dict_r = self.conv_RGB.state_dict()
        state_dict_r = {k:v for k, v in save_model.items() if k in model_dict_r.keys()}
        model_dict_r.update(state_dict_r)
        self.conv_RGB.load_state_dict(model_dict_r)
        print(f"RGB Loading pre_model ${pre_model}")

        save_model = torch.load(pre_model)
        model_dict_d = self.conv_depth.state_dict()
        state_dict_d = {k: v for k, v in save_model.items() if k in model_dict_d.keys()}
        model_dict_d.update(state_dict_d)
        self.conv_depth.load_state_dict(model_dict_d)
        print(f"Depth Loading pre_model ${pre_model}")

if __name__ == "__main__":
    model = CoNet()

    def print_network(model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print("The number of parameters:{}M".format(num_params / 1000000))

    model.train()
    depth = torch.randn(5, 3, 256, 256)
    input = torch.randn(5, 3, 256, 256)
    gts = torch.randn(5, 1, 256, 256)
    # model.load_pre('/home/map/Alchemist/COA/COA_RGBD_SOD/al/Pretrain/segformer.b4.512x512.ade.160k.pth')
    flops, params = profile(model, inputs=(input, depth))
    print("the number of Flops {} G ".format(flops / 1e9))
    print("the number of Parameter {}M ".format(params / 1e6)) #1048576

    print_network(model, 'ccc')

    out = model(input, depth)
    for i in range(len(out)):
        print(out[i].shape)

