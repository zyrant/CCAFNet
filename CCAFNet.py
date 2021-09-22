import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

class Separable_conv(nn.Module):
    def __init__(self, inp, oup):
        super(Separable_conv, self).__init__()

        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=3, stride=1, padding=1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(inp, oup, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)


model = models.vgg16_bn(pretrained=True)
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class vgg_rgb(nn.Module):
    def __init__(self, pretrained=True):
        super(vgg_rgb, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # first model 224*24*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # [:6]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),  # second model 112*112*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # [6:13]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),  # third model 56*56*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # [13:23]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, 3, 1, 1),  # forth model 28*28*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [13:33]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, 3, 1, 1),  # fifth model 14*14*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [33:43]
        )

        if pretrained:
            pretrained_vgg = model_zoo.load_url(model_urls['vgg16_bn'])
            model_dict = {}
            state_dict = self.state_dict()
            for k, v in pretrained_vgg.items():
                if k in state_dict:
                    model_dict[k] = v
                    # print(k, v)

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, rgb):
        A1 = self.features[:6](rgb)
        A2 = self.features[6:13](A1)
        A3 = self.features[13:23](A2)
        A4 = self.features[23:33](A3)
        A5 = self.features[33:43](A4)
        return A1, A2, A3, A4, A5


class vgg_depth(nn.Module):
    def __init__(self, pretrained=True):
        super(vgg_depth, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # first model 224*224*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # [:6]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),  # second model 112*112*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # [6:13]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),  # third model 56*56*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # [13:23]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, 3, 1, 1),  # forth model 28*28*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [13:33]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, 3, 1, 1),  # fifth model 14*14*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [33:43]
        )

        if pretrained:
            pretrained_vgg = model_zoo.load_url(model_urls['vgg16_bn'])
            model_dict = {}
            state_dict = self.state_dict()
            for k, v in pretrained_vgg.items():
                if k in state_dict:
                    model_dict[k] = v
                    # print(k, v)

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, thermal):
        A1_d = self.features[:6](thermal)
        A2_d = self.features[6:13](A1_d)
        A3_d = self.features[13:23](A2_d)
        A4_d = self.features[23:33](A3_d)
        A5_d = self.features[33:43](A4_d)
        return A1_d, A2_d, A3_d, A4_d, A5_d


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class Spatical_Fuse_attention3_GHOST(nn.Module):  # 最终为rgb  rgb, y为depth 加入恒等变化
    def __init__(self, in_channels,):
        super(Spatical_Fuse_attention3_GHOST, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, 3, 1, 1)
        self.active = Hsigmoid()

    def forward(self, x, y):
        input_y = self.conv(y)
        input_y = self.active(input_y)
        # return input_y
        return x + x * input_y

class Channel_Fuse_attention2(nn.Module):  # 最终为depth  x为depth, y为rgb 加入恒等变化
    def __init__(self, channel, reduction=4):
        super(Channel_Fuse_attention2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x, y):
        b, c, _, _ = x.size()
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x + x * y.expand_as(x)


class Gatefusion3(nn.Module):
    def __init__(self, channel):
        super(Gatefusion3, self).__init__()
        self.channel = channel
        self.gate = nn.Sigmoid()

    def forward(self, x, y, fusion_up):
        first_fusion = torch.cat((x, y), dim=1)
        gate_fusion = self.gate(first_fusion)
        gate_fusion = torch.split(gate_fusion, self.channel, dim=1)
        fusion_x = gate_fusion[0] * x + x
        fusion_y = gate_fusion[1] * y + y
        fusion = fusion_x + fusion_y
        fusion = torch.abs((fusion - fusion_up)) * fusion + fusion
        return fusion

class Gatefusion3_fusionup(nn.Module):
    def __init__(self, channel):
        super(Gatefusion3_fusionup, self).__init__()
        self.channel = channel
        self.gate = nn.Sigmoid()

    def forward(self, x, y):
        first_fusion = torch.cat((x, y), dim=1)
        gate_fusion = self.gate(first_fusion)
        gate_fusion = torch.split(gate_fusion, self.channel, dim=1)
        fusion_x = gate_fusion[0] * x + x
        fusion_y = gate_fusion[1] * y + y
        fusion = fusion_x + fusion_y
        return fusion

class CCAFNet(nn.Module):
    def __init__(self, ):
        super(CCAFNet, self).__init__()
        # rgb,depth encode
        self.rgb_pretrained = vgg_rgb()
        self.depth_pretrained = vgg_depth()

        # rgb Fuse_model
        self.SAG1 = Spatical_Fuse_attention3_GHOST(64)
        self.SAG2 = Spatical_Fuse_attention3_GHOST(128)
        self.SAG3 = Spatical_Fuse_attention3_GHOST(256)

        # depth Fuse_model
        self.CAG4 = Channel_Fuse_attention2(512)
        self.CAG5 = Channel_Fuse_attention2(512)

        self.gatefusion5 = Gatefusion3_fusionup(512)
        self.gatefusion4 = Gatefusion3(512)
        self.gatefusion3 = Gatefusion3(256)
        self.gatefusion2 = Gatefusion3(128)
        self.gatefusion1 = Gatefusion3(64)


        # Upsample_model
        self.upsample1 = nn.Sequential(nn.Conv2d(288, 144, 3, 1, 1),nn.BatchNorm2d(144),nn.ReLU())
        self.upsample2 = nn.Sequential(nn.Conv2d(448, 224,3,1,1),nn.BatchNorm2d(224),nn.ReLU(),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.upsample3 = nn.Sequential(nn.Conv2d(640, 320,3,1,1),nn.BatchNorm2d(320),nn.ReLU(),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.upsample4 = nn.Sequential(nn.Conv2d(768, 384,3,1,1),nn.BatchNorm2d(384),nn.ReLU(),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.upsample5 = nn.Sequential(nn.Conv2d(512, 256,3,1,1),nn.BatchNorm2d(256),nn.ReLU(),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))

        # duibi
        self.upsample5_4 = nn.Sequential(nn.Conv2d(512, 512,3,1,1),nn.BatchNorm2d(512),nn.ReLU(),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.upsample4_3 = nn.Sequential(nn.Conv2d(768, 256,3,1,1),nn.BatchNorm2d(256),nn.ReLU(),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.upsample3_2 = nn.Sequential(nn.Conv2d(640, 128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.upsample2_1 = nn.Sequential(nn.Conv2d(448, 64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.conv = nn.Conv2d(144, 1, 1)
        self.conv2 = nn.Conv2d(224, 1, 1)
        self.conv3 = nn.Conv2d(320, 1, 1)
        self.conv4 = nn.Conv2d(384, 1, 1)
        self.conv5 = nn.Conv2d(256, 1, 1)

    def forward(self, rgb, depth):
        # rgb
        A1, A2, A3, A4, A5 = self.rgb_pretrained(rgb)
        # depth
        A1_d, A2_d, A3_d, A4_d, A5_d = self.depth_pretrained(depth)

        SAG1_R  = self.SAG1(A1, A1_d)
        SAG2_R = self.SAG2(A2, A2_d)
        SAG3_R = self.SAG3(A3, A3_d)

        CAG5_D = self.CAG5(A5_d, A5)
        CAG4_D = self.CAG4(A4_d, A4)

        F5 = self.gatefusion5(A5, CAG5_D)
        F5_UP = self.upsample5_4(F5)
        F5 = self.upsample5(F5)  # 14*14
        F4 = self.gatefusion4(A4, CAG4_D, F5_UP)
        F4 = torch.cat((F4, F5), dim=1)
        F4_UP = self.upsample4_3(F4)
        F4 = self.upsample4(F4)  # 28*28
        F3 = self.gatefusion3(SAG3_R, A3_d, F4_UP)
        F3 = torch.cat((F3, F4), dim=1)
        F3_UP = self.upsample3_2(F3)
        F3 = self.upsample3(F3)  # 56*56
        F2 = self.gatefusion2(SAG2_R, A2_d, F3_UP)
        F2 = torch.cat((F2, F3), dim=1)
        F2_UP = self.upsample2_1(F2)
        F2 = self.upsample2(F2)  # 112*112
        F1 = self.gatefusion1(SAG1_R, A1_d, F2_UP)
        F1 = torch.cat((F1, F2), dim=1)
        F1 = self.upsample1(F1)  # 224*224
        out = self.conv(F1)

        out5 = self.conv5(F5)
        out4 = self.conv4(F4)
        out3 = self.conv3(F3)
        out2 = self.conv2(F2)

        if self.training:
            return out, out2, out3, out4, out5
        return out




if __name__=='__main__':

    # model = ghost_net()
    # model.eval()
    model = CCAFNet()
    rgb = torch.randn(1, 3, 224, 224)
    depth = torch.randn(1, 3, 224, 224)
    out = model(rgb,depth)
    for i in out:
        print(i.shape)
