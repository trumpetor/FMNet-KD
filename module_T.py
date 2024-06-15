import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


# class Attention(nn.Module):
#     def __init__(self,dim,num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,proj_drop=0.):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.qkv = nn.Sequential(nn.Linear(dim, dim//8, bias = qkv_bias), nn.Linear(dim//8, dim*3))
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, r):
#         x = r.flatten(2).transpose(1, 2)
#         B,N,C = x.shape
#         qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C // self.num_heads).permute(2,0,3,1,4)
#         q,k,v = qkv[0],qkv[1],qkv[2]
#         attn = (q @ k.transpose(-2,-1)) * self.scale
#         attn = attn.softmax(dim = -1)
#         x = (attn @ v).transpose(1,2).reshape(B,N,C)
#         x = self.proj(x) + x
#         x = self.proj_drop(x)
#         x = x.view(B, int(np.sqrt(N)), int(np.sqrt(N)), -1).permute(0, 3, 1, 2).contiguous()
#         x = x + r
#         return x





class MSFA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MSFA, self).__init__()
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Sequential(DepthWiseConv(in_ch, out_ch), nn.BatchNorm2d(out_ch), nn.ReLU())


    def forward(self, fuse_high, fuse_low):
        fuse_high = self.up2(fuse_high)
        fe_decode = self.conv(fuse_high) + fuse_low
        return fe_decode

# Cascaded Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.channel = [64, 128, 320, 640]
        self.cfm12 = MSFA(self.channel[3], self.channel[2])
        self.cfm23 = MSFA(self.channel[2], self.channel[1])
        self.cfm34 = MSFA(self.channel[1], self.channel[0])
        self.conv1_2 = nn.Sequential(DepthWiseConv(self.channel[0], self.channel[1]), nn.BatchNorm2d(self.channel[1]), nn.ReLU())
        self.conv1_3 = nn.Sequential(DepthWiseConv(self.channel[0], self.channel[2]), nn.BatchNorm2d(self.channel[2]), nn.ReLU())
        self.conv1_4 = nn.Sequential(DepthWiseConv(self.channel[0], self.channel[3]), nn.BatchNorm2d(self.channel[3]), nn.ReLU())



    def forward(self, fuse4, fuse3, fuse2, fuse1, iter=None):
        if iter is not None:
            out_fuse4 = F.interpolate(iter, size=(8, 8), mode='bilinear')
            out_fuse4 = self.conv1_4(out_fuse4)
            fuse4 = out_fuse4 + fuse4

            out_fuse3 = F.interpolate(iter, size=(16, 16), mode='bilinear')
            out_fuse3 = self.conv1_3(out_fuse3)
            fuse3 = out_fuse3 + fuse3

            out_fuse2 = F.interpolate(iter, size=(32, 32), mode='bilinear')
            out_fuse2 = self.conv1_2(out_fuse2)
            fuse2 = out_fuse2 + fuse2

            fuse1 = iter + fuse1

            out43 = self.cfm12(fuse4, fuse3)
            out432 = self.cfm23(out43, fuse2)
            out4321 = self.cfm34(out432, fuse1)
            return out4321
        else:
            out43 = self.cfm12(fuse4, fuse3)  # [b,256,16,16]
            out432 = self.cfm23(out43, fuse2)  # [b,128,32,32]
            out4321 = self.cfm34(out432, fuse1)  # [b,64,64,64]
            return out4321, out432, out43


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)
        self.point_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0, groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)

        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.fc1 = DepthWiseConv(in_features, hidden_features)
        self.fc2 = DepthWiseConv(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class RFEM(nn.Module):
    def __init__(self, channels1, channels2):
        super(RFEM, self).__init__()
        self.channels1 = channels1
        self.channels2 = channels2
        self.rgb_channel_attention = ChannelAttention(channels1)
        self.rd_spatial_attention = SpatialAttention()
        self.rgb_spatial_attention = SpatialAttention()
        self.rgb_context = nn.Sequential(DepthWiseConv(channels2 + channels1, channels1), nn.BatchNorm2d(channels1), nn.ReLU())
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, r1, r2):
        if self.channels1 == self.channels2:
            mul_fuse = r1 * r2
        else:
            mul_fuse = self.rgb_context(torch.cat([self.up(r2), r1], dim=1))
        sa = self.rd_spatial_attention(mul_fuse)
        r_f = mul_fuse * sa
        r_f = r1 + r_f
        r_ca = self.rgb_channel_attention(r_f)
        r_out = r_f * r_ca + r1
        return r_out

class DFEM(nn.Module):
    def __init__(self, channels1, channels2):
        super(DFEM, self).__init__()
        self.channels1 = channels1
        self.channels2 = channels2
        self.depth_spatial_attention = SpatialAttention()
        self.depth_channel_attention = ChannelAttention(channels1)
        self.rd_spatial_attention = SpatialAttention()
        self.depth_context = nn.Sequential(DepthWiseConv(channels2 + channels1, channels1), nn.BatchNorm2d(channels1), nn.ReLU())
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, d1, d2):
        if self.channels1 == self.channels2:
            mul_fuse = d1 * d2
        else:
            mul_fuse = self.depth_context(torch.cat([self.up(d2), d1], dim=1))
        sa = self.rd_spatial_attention(mul_fuse)
        d_f = mul_fuse * sa
        d_f = d1 + d_f
        d_ca = self.depth_channel_attention(d_f)
        d_out = d_f * d_ca + d1
        return d_out


# class CMFM(nn.Module):
#     def __init__(self, channels1, channels2):
#         super(CMFM, self).__init__()
#         self.dfem = DFEM(channels1, channels2)
#         self.rfem = RFEM(channels1, channels2)
#         self.channels1 = channels1
#         self.channels2 = channels2
#         self.conv = nn.Sequential(DepthWiseConv(channels1 * 2, channels1), nn.BatchNorm2d(channels1), nn.ReLU())
#
#     def forward(self, r, d, rd):
#         fr = self.rfem(r, rd)
#         fd = self.dfem(d, rd)
#         mul_fea = fr * fd
#         add_fea = fr + fd
#         fuse_fea = torch.cat([mul_fea, add_fea], dim=1)
#         fuse_fea = self.conv(fuse_fea)
#         return fuse_fea, fr, fd
#
# class CMFM4(nn.Module):
#     def __init__(self, channels1, channels2):
#         super(CMFM4, self).__init__()
#         self.dfem = DFEM(channels1, channels2)
#         self.rfem = RFEM(channels1, channels2)
#         self.channels1 = channels1
#         self.channels2 = channels2
#         self.conv = nn.Sequential(DepthWiseConv(channels1 * 2, channels1), nn.BatchNorm2d(channels1), nn.ReLU())
#
#
#     def forward(self, r, d):
#         fr = self.rfem(r, d)
#         fd = self.dfem(d, r)
#         mul_fea = fr * fd
#         add_fea = fr + fd
#         fuse_fea = torch.cat([mul_fea, add_fea], dim=1)
#         fuse_fea = self.conv(fuse_fea)
#         return fuse_fea, fr, fd


class block(nn.Module):
    def __init__(self, dim1, dim2):
        super(block, self).__init__()
        self.rfem = RFEM(dim1, dim2)
        self.dfem = DFEM(dim1, dim2)
        self.cfmm = CFMM(dim1)
    def forward(self, rgb, depth, rd):
        fr = self.rfem(rgb, rd)
        fd = self.dfem(depth, rd)
        F, a = self.cfmm(fr, fd)
        return F, a, fr, fd

class block4(nn.Module):
    def __init__(self, dim1, dim2):
        super(block4, self).__init__()
        self.rfem = RFEM(dim1, dim2)
        self.dfem = DFEM(dim1, dim2)
        self.cfmm = CFMM(dim1)
    def forward(self, rgb, depth):
        fr = self.rfem(rgb, depth)
        fd = self.dfem(depth, rgb)
        F, a = self.cfmm(fr, fd)
        return F, a, fr, fd

class CFMM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(DepthWiseConv(dim * 2, dim), nn.BatchNorm2d(dim), nn.ReLU())
        self.frd_fc_h = nn.Conv2d(dim, dim, 1, 1)
        self.frd_fc_w = nn.Conv2d(dim, dim, 1, 1)
        self.frd_fc_c = nn.Conv2d(dim, dim, 1, 1)

        self.frd_tfc_h = nn.Conv2d(2 * dim, dim, (1, 7), stride=1, padding=(0, 7 // 2), groups=dim, bias=False)
        self.frd_tfc_w = nn.Conv2d(2 * dim, dim, (7, 1), stride=1, padding=(7 // 2, 0), groups=dim, bias=False)
        self.frd_proj = nn.Sequential(nn.Conv2d(dim * 4, dim, 1, 1), nn.BatchNorm2d(dim), nn.ReLU())

        self.frd_theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1), nn.BatchNorm2d(dim), nn.ReLU())
        self.frd_theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1), nn.BatchNorm2d(dim), nn.ReLU())
        self.frd_reweight = Mlp(dim * 3, dim, dim * 3)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fr, fd):
        mul_fea = fr * fd
        add_fea = fr + fd
        fuse_fea = torch.cat([mul_fea, add_fea], dim=1)
        frd = self.conv(fuse_fea)
        B, C, H, W = frd.shape
        frd_theta_h = self.frd_theta_h_conv(frd)
        frd_theta_w = self.frd_theta_w_conv(frd)

        frd_h = self.frd_fc_h(frd)
        frd_w = self.frd_fc_w(frd)

        frd_h = torch.cat([frd_h * torch.cos(frd_theta_h), frd_h * torch.sin(frd_theta_h)], dim=1)
        frd_w = torch.cat([frd_w * torch.cos(frd_theta_w), frd_w * torch.sin(frd_theta_w)], dim=1)

        h_frd = self.frd_tfc_h(frd_h)
        w_frd = self.frd_tfc_w(frd_w)
        c_frd = self.frd_fc_c(frd)

        a = torch.cat([h_frd, w_frd, c_frd], dim=1)
        a = self.frd_reweight(a).reshape(B, C, 3, H, W).permute(2, 0, 1, 3, 4).softmax(dim=0)

        r = h_frd * a[0] + w_frd * a[1] + c_frd * a[2]
        r = self.frd_proj(torch.cat([frd, r, fr, fd], dim=1))
        return r, a

if __name__ == "__main__":


    def print_network(model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print("The number of parameters:{}M".format(num_params / 1000000))

    model4 = block(256, 512)
    # model3 = FMDM(256, 16)
    # model2 = FMDM(128, 32)
    # model1 = FMDM(64, 64)
    d_convs4 = nn.ModuleList([nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512) for temp in [1,2,3,4]])
    depth = torch.randn(5, 256, 16, 16)
    input = torch.randn(5, 256, 16, 16)
    rd = torch.randn(5, 512, 8, 8)
    out = model4(input, depth, rd)
    # print(out.shape)
    for i in range(len(out)):
        print("out shape", out[i].shape)
    # flops, params = profile(model, inputs=(input))
    # print("the number of Flops {} G ".format(flops / 1e9))
    # print("the number of Parameter {}M ".format(params / 1e6))  # 1048576
    print_network(model4, 'FMDM4')