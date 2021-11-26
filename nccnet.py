import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__   
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)

def once_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.Tanh(),
        nn.BatchNorm2d(out_channels)
    )


class Residual_block(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        
        self.conv1 = once_conv(in_channels, out_channels)
        
        self.conv2 = once_conv(out_channels, out_channels)
        self.conv3 = once_conv(out_channels, out_channels)
        self.conv4 = once_conv(out_channels, out_channels)
        
        self.outconv = once_conv(out_channels+out_channels, out_channels)

    def forward(self, x):
        c = self.conv1(x)
        
        x = self.conv2(c)
        x = self.conv3(x)
        x = self.conv4(x)

        x = torch.cat([x, c], dim=1)
        out = self.outconv(x)

        return out

class NCCNet(nn.Module):

    def __init__(self, out_channels=1) -> None:
        super().__init__()
        self.out_channels = out_channels

        # 符合NCCNet文章的通道数
        self.rescov_d1 = Residual_block(1, 8)
        self.rescov_d2 = Residual_block(8, 16)
        self.rescov_d3 = Residual_block(16, 32)
        self.rescov_d4 = Residual_block(32, 64)

        self.maxpool = nn.MaxPool2d(2)

        # NCCNet文章中使用resizeConv模块 先resize再进行卷积 卷积参数和UNet相同
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.uconv_u3 = once_conv(64, 32)
        self.uconv_u2 = once_conv(32, 16)
        self.uconv_u1 = once_conv(16, 8)

        self.rescov_u3 = Residual_block(32 + 32, 32)
        self.rescov_u2 = Residual_block(16 + 16, 16)
        self.rescov_u1 = Residual_block(8 + 8, 8)

        self.rescov_out = Residual_block(8, self.out_channels)


    def forward_once(self, x):
        conv1 = self.rescov_d1(x)
        x = self.maxpool(conv1)

        conv2 = self.rescov_d2(x)
        x = self.maxpool(conv2)

        conv3 = self.rescov_d3(x)
        x = self.maxpool(conv3)

        x = self.rescov_d4(x)

        x = self.upsample(x)
        x = self.uconv_u3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.rescov_u3(x)

        x = self.upsample(x)
        x = self.uconv_u2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.rescov_u2(x)

        x = self.upsample(x)
        x = self.uconv_u1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.rescov_u1(x)

        out = self.rescov_out(x)
        return out

    def forward(self, template, source):
        tout = self.forward_once(template)
        sout = self.forward_once(source)
        return tout, sout


# 2021.10.12
#   尝试添加注意力模块  并作为输出
# 2021.11.11
#   新建别的模型文件
