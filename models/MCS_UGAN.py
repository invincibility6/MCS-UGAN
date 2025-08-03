import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, 3, 1, padding=1, padding_mode='zeros')
        self.t = nn.Sequential(self.conv, nn.InstanceNorm2d(output_channel), nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.t(x)
        return x1


class UNetDown(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(UNetDown, self).__init__()
        self.DownConv = nn.Conv2d(input_channel, output_channel, 3, 2, padding=1, padding_mode='zeros')
        self.down = nn.Sequential(self.DownConv, nn.InstanceNorm2d(output_channel), nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.down(x)
        return x1


class UNetUp(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(UNetUp, self).__init__()
        self.UpConv = nn.ConvTranspose2d(input_channel, output_channel, 4, 2, padding=1, padding_mode='zeros')
        self.up = nn.Sequential(self.UpConv, nn.InstanceNorm2d(output_channel), nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.up(x)
        return x1


class ResNet(nn.Module):
    def __init__(self, dil):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(256, 256, 3, 1, padding=dil, padding_mode='zeros', dilation=dil)
        self.res = nn.Sequential(self.conv, nn.InstanceNorm2d(256), nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.res(x)
        x2 = x + x1
        return x2


class Final(nn.Module):
    def __init__(self):
        super(Final, self).__init__()
        self.conv = nn.Conv2d(64, 3, 3, 1, padding=1, padding_mode='zeros')
        self.t = nn.Sequential(self.conv, nn.Tanh())

    def forward(self, x):
        x1 = self.t(x)
        return x1


class MRDC(nn.Module):
    def __init__(self):
        super(MRDC, self).__init__()
        self.convA = nn.Conv2d(32, 32, 3, 1, padding=1, padding_mode='zeros', dilation=1)
        self.convB = nn.Conv2d(32, 32, 3, 1, padding=3, padding_mode='zeros', dilation=3)
        self.convC = nn.Conv2d(32, 32, 3, 1, padding=5, padding_mode='zeros', dilation=5)
        self.conv = nn.Conv2d(64, 32, 1, 1, padding=0)
        self.A = nn.Sequential(self.convA, nn.ReLU(inplace=True), self.convA)
        self.B = nn.Sequential(self.convB, nn.ReLU(inplace=True), self.convB)
        self.C = nn.Sequential(self.convC, nn.ReLU(inplace=True), self.convC)
    def forward(self, x):
        A = self.A(x) + x
        B = self.B(x) + x
        C = self.C(x) + x
        t1 = torch.cat((B, C), 1)
        t2 = self.conv(t1)
        t3 = torch.cat((A, t2), 1)
        f = self.conv(t3)
        return f


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)



def rgb2yuv(rgb):
    Y = 0.2990 * rgb[:, 0, :, :] + 0.5870 * rgb[:, 1, :, :] + 0.1140 * rgb[:, 2, :, :]
    Y1 = Y
    Y = Y.reshape(Y.shape[0], 1, Y.shape[1], Y.shape[2])
    U = (rgb[:, 2, :, :] - Y1) / 1.772
    U = U.reshape(U.shape[0], 1, U.shape[1], U.shape[2])
    V = 0.5 * rgb[:, 0, :, :] - 0.419 * rgb[:, 1, :, :] - 0.081 * rgb[:, 2, :, :]
    V = V.reshape(V.shape[0], 1, V.shape[1], V.shape[2])
    yuv = torch.cat([Y, U, V], 1)
    return yuv


def yuv2rgb(yuv):
    R = yuv[:, 0, :, :] + 1.4075 * yuv[:, 2, :, :]
    R = R.reshape(R.shape[0], 1, R.shape[1], R.shape[2])
    G = yuv[:, 0, :, :] - 0.3455 * (yuv[:, 1, :, :]) - 0.7169 * (yuv[:, 2, :, :])
    G = G.reshape(G.shape[0], 1, G.shape[1], G.shape[2])
    B = yuv[:, 0, :, :] + 1.779 * (yuv[:, 1, :, :])
    B = B.reshape(B.shape[0], 1, B.shape[1], B.shape[2])
    rgb = torch.cat([R, G, B], 1)
    return rgb


class test6_Generator(nn.Module):
    def __init__(self):
        super(test6_Generator, self).__init__()
        # encoding layers
        self.conv_s1 = nn.Conv2d(1, 32, 3, 1, padding=1, padding_mode='zeros')
        self.conv_s2 = nn.Conv2d(32, 32, 3, 1, padding=1, padding_mode='zeros')
        self.conv_s3 = nn.Conv2d(32, 32, 3, 1, padding=1, padding_mode='zeros')
        self.conv_e1 = nn.Conv2d(32, 32, 3, 1, padding=1, padding_mode='zeros')
        self.conv_e2 = nn.Conv2d(32, 32, 3, 1, padding=1, padding_mode='zeros')
        self.conv_e3 = nn.Conv2d(32, 1, 3, 1, padding=1, padding_mode='zeros')
        self.convS = nn.Sequential(self.conv_s1, self.conv_s2, self.conv_s3)
        self.convE = nn.Sequential(self.conv_e1, self.conv_e2, self.conv_e3)
        self.MRDC1 = MRDC()
        self.EMA1 = EMA(32)
        self.MRDC2 = MRDC()
        self.EMA2 = EMA(32)

        self.conv1 = Conv(3, 64)
        self.conv2 = Conv(64, 128)
        self.DownConv3 = UNetDown(128, 128)
        self.conv4 = Conv(128, 128)
        self.conv5 = Conv(128, 256)
        self.DownConv6 = UNetDown(256, 256)
        self.res1 = ResNet(2)
        self.res2 = ResNet(2)
        self.res3 = ResNet(4)
        self.res4 = ResNet(4)
        self.UpConv7 = UNetUp(512, 256)
        self.conv8 = Conv(256, 128)
        self.conv9 = Conv(128, 128)
        self.UpConv10 = UNetUp(256, 128)
        self.conv11 = Conv(128, 64)
        self.conv12 = Final()

    def forward(self, x):
        img_yuv = rgb2yuv(x)
        img_yuv = torch.split(img_yuv, 1, 1)
        S = self.convS(img_yuv[0])
        F0 = self.MRDC1(S)
        G0 = self.EMA1(F0)
        F1 = self.MRDC2(G0)
        G1 = self.EMA2(F1)
        N = G1 - S
        img_y = self.convE(N)
        img = torch.cat((img_y, img_yuv[1], img_yuv[2]), 1)
        img_rgb = yuv2rgb(img)
        x1 = self.conv2(self.conv1(img_rgb))
        d1 = self.DownConv3(x1)
        x2 = self.conv5(self.conv4(d1))
        d2 = self.DownConv6(x2)
        r1 = self.res2(self.res1(d2))
        r2 = self.res4(self.res3(r1))
        u1 = self.UpConv7(torch.cat((r2, d2), 1))
        x3 = self.conv9(self.conv8(u1))
        u2 = self.UpConv10(torch.cat((x3, d1), 1))
        out = self.conv12(self.conv11(u2))
        return out


class Discriminator(nn.Module):
    """ A 4-layer Markovian discriminator
    """
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            #Returns downsampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
