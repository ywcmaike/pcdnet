import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


def downconv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=2, padding=(kernel_size-1)//2),
        nn.BatchNorm2d(out_planes),

        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size,
                  padding=(kernel_size-1)//2),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size-1)//2),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3,
                           stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


def output(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3,
                           stride=2, padding=1, output_padding=1),
        nn.Sigmoid()
    )

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()

        vgg_down_planes = [16, 32, 64, 128, 256, 512]
        self.layer0_1 = conv(3, vgg_down_planes[0])
        self.layer0_2 = conv(vgg_down_planes[0], vgg_down_planes[0])

        self.layer1_1 = conv(vgg_down_planes[0], vgg_down_planes[1], stride=2)
        self.layer1_2 = conv(vgg_down_planes[1], vgg_down_planes[1])
        self.layer1_3 = conv(vgg_down_planes[1], vgg_down_planes[1])

        self.layer2_1 = conv(vgg_down_planes[1], vgg_down_planes[2], stride=2)
        self.layer2_2 = conv(vgg_down_planes[2], vgg_down_planes[2])
        self.layer2_3 = conv(vgg_down_planes[2], vgg_down_planes[2])

        self.layer3_1 = conv(vgg_down_planes[2], vgg_down_planes[3], stride=2)
        self.layer3_2 = conv(vgg_down_planes[3], vgg_down_planes[3])
        self.layer3_3 = conv(vgg_down_planes[3], vgg_down_planes[3])

        self.layer4_1 = conv(vgg_down_planes[3], vgg_down_planes[4], kernel_size=5, stride=2)
        self.layer4_2 = conv(vgg_down_planes[4], vgg_down_planes[4])
        self.layer4_3 = conv(vgg_down_planes[4], vgg_down_planes[4])

        self.layer5_1 = conv(vgg_down_planes[4], vgg_down_planes[5], kernel_size=5, stride=2)
        self.layer5_2 = conv(vgg_down_planes[5], vgg_down_planes[5])
        self.layer5_3 = conv(vgg_down_planes[5], vgg_down_planes[5])
        self.layer5_4 = conv(vgg_down_planes[5], vgg_down_planes[5])

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        x = self.layer0_1(img)
        x = self.layer0_2(x)
        feat0_2 = x

        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x = self.layer1_3(x)
        feat1_3 = x
        feat1_3 = torch.repeat_interleave(feat1_3, 2, dim=-2)
        feat1_3 = torch.repeat_interleave(feat1_3, 2, dim=-1)

        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)  # 64
        feat2_3 = x
        feat2_3 = torch.repeat_interleave(feat2_3, 4, dim=-2)
        feat2_3 = torch.repeat_interleave(feat2_3, 4, dim=-1)

        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)  # 128
        feat3_3 = x
        feat3_3 = torch.repeat_interleave(feat3_3, 8, dim=-2)
        feat3_3 = torch.repeat_interleave(feat3_3, 8, dim=-1)

        x = self.layer4_1(x)
        x = self.layer4_2(x)
        x = self.layer4_3(x)  # 256
        feat4_3 = x
        feat4_3 = torch.repeat_interleave(feat4_3, 16, dim=-2)
        feat4_3 = torch.repeat_interleave(feat4_3, 16, dim=-1)

        x = self.layer5_1(x)
        x = self.layer5_2(x)
        x = self.layer5_3(x)
        x = self.layer5_4(x)  # 512
        feat5_4 = x
        feat5_4 = torch.repeat_interleave(feat5_4, 32, dim=-2)
        feat5_4 = torch.repeat_interleave(feat5_4, 32, dim=-1)

        feat = torch.cat([feat0_2, feat1_3, feat2_3, feat3_3, feat4_3, feat5_4], dim=1)
        return feat, feat2_3, feat3_3, feat4_3, feat5_4     # 1008 64 128 256 512


if __name__ == '__main__':
    img = torch.randn(8, 3, 224, 224)
    print(img.shape)
    vgg = VGGNet()
    feat = vgg(img)
    print(feat.shape)
