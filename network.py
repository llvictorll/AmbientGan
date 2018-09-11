import torch.nn as nn
import torch.nn.functional as F


class NetG(nn.Module):
    def __init__(self):
        super(NetG, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)

        self.convT1 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False)
        self.convT2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.convT3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.convT4 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)

        self.convT5 = nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False)
        self.convT6 = nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False)

        self.conv_drop2d = nn.Dropout2d()

        self.batch1 = nn.BatchNorm2d(64, affine=False)
        self.batch2 = nn.BatchNorm2d(128, affine=False)
        self.batch3 = nn.BatchNorm2d(256, affine=False)
        self.batch4 = nn.BatchNorm2d(256, affine=False)
        self.batch5 = nn.BatchNorm2d(128, affine=False)
        self.batch6 = nn.BatchNorm2d(64, affine=False)
        self.batch7 = nn.BatchNorm2d(32, affine=False)
        self.batch8 = nn.BatchNorm2d(16, affine=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        x1 = F.relu(self.batch1(self.conv1(x)))
        x2 = F.relu(self.batch2(self.conv2(x1)))
        x2 = self.conv_drop2d(x2)
        x3 = F.relu(self.batch3(self.conv3(x2)))
        x4 = F.relu(self.conv4(x3))
        """
        x = F.relu(self.batch4(self.convT1(x)))
        x = F.relu(self.batch5(self.convT2(x)))
        x = self.conv_drop2d(x)
        x = F.relu(self.batch6(self.convT3(x)))
        x = F.relu(self.batch7(self.convT4(x)))
        x = self.conv_drop2d(x)
        x = F.relu(self.batch8(self.convT5(x)))
        x = self.tanh(self.convT6(x))
        return x


class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5 = nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.batch1 = nn.BatchNorm2d(128, affine=False)
        self.batch2 = nn.BatchNorm2d(256, affine=False)
        self.batch3 = nn.BatchNorm2d(256, affine=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.batch1(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.batch2(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.batch3(self.conv4(x)), 0.2)
        x = self.conv5(x)
        return x.squeeze()


class NetD_super(nn.Module):
    def __init__(self, mode='low'):
        super(NetD_super, self).__init__()
        self.mode = mode
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv4bis = nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5bis = nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv6bis = nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1, bias=False)

        self.batch1 = nn.BatchNorm2d(128, affine=False)
        self.batch2 = nn.BatchNorm2d(256, affine=False)
        self.batch3 = nn.BatchNorm2d(512, affine=False)
        self.batch4 = nn.BatchNorm2d(512, affine=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.batch1(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.batch2(self.conv3(x)), 0.2)
        if self.mode == 'verylow':
            x = self.conv4bis(x)
        if self.mode == 'low':
            x = F.leaky_relu(self.batch3(self.conv4(x)), 0.2)
            x = self.conv5bis(x)
        elif self.mode == 'high':
            x = F.leaky_relu(self.batch3(self.conv4(x)), 0.2)
            x = F.leaky_relu(self.batch4(self.conv5(x)), 0.2)
            x = self.conv6bis(x)
        return x.squeeze()


class NetG_super(nn.Module):
    def __init__(self):
        super(NetG_super, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)

        self.convT1 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.convT2 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.convT3 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
        self.convT4 = nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False)

        self.conv_drop2d = nn.Dropout2d()

        self.batch1 = nn.BatchNorm2d(64, affine=False)
        self.batch2 = nn.BatchNorm2d(128, affine=False)
        self.batch3 = nn.BatchNorm2d(128, affine=False)
        self.batch4 = nn.BatchNorm2d(64, affine=False)
        self.batch5 = nn.BatchNorm2d(32, affine=False)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = F.relu(self.batch1(self.conv1(x)))
        x2 = F.relu(self.batch2(self.conv2(x1)))
        x2 = self.conv_drop2d(x2)

        x3 = F.relu(self.conv3(x2))
        out = F.relu(self.batch3(self.convT1(x3)))
        out = F.relu(self.batch4(self.convT2(out + x2)))
        out = self.conv_drop2d(out)
        #out = F.relu(self.batch5(self.convT3(out + x1)))

        out = self.tanh(self.convT3(out))
        return out

class NetG_srgan(nn.Module):
    def __init__(self):
        super(NetG_srgan, self).__init__()
        self.convInit = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.convRes1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convRes2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convRes3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convRes4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convRes5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.convResInit = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)

        self.convOut = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)

        self.pix_shuf1 = nn.PixelShuffle(2)
        self.pix_shuf2 = nn.PixelShuffle(2)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        self.prelu5 = nn.PReLU()
        self.prelu6 = nn.PReLU()

        self.batch = nn.BatchNorm2d(64, affine=False)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.prelu1(self.convInit(x))
        xres1 = self.batch(self.convRes2(self.prelu2(self.batch(self.convRes1(x)))))+x
        xres2 = self.batch(self.convRes3(self.prelu3(self.batch(self.convRes3(xres1)))))+xres1
        xres3 = self.batch(self.convRes5(self.prelu4(self.batch(self.convRes4(xres2)))))+xres2
        x = self.batch(self.convResInit(xres3))+x
        print(x.size())
        x = self.prelu5(self.pix_shuf1(self.conv1(x)))
        print(x.size())
        x = self.prelu6(self.pix_shuf2(self.conv2(x)))
        print(x.size())
        x = self.tanh(self.convOut(x))
        print(x.size())
        return x


class NetD_patch(nn.Module):
    def __init__(self):
        super(NetD_patch, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv6 = nn.Conv2d(512, 256, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv7 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.batch1 = nn.BatchNorm2d(128, affine=False)
        self.batch2 = nn.BatchNorm2d(256, affine=False)
        self.batch3 = nn.BatchNorm2d(512, affine=False)

    def forward(self, x):
        #print(x.size())
        x = F.leaky_relu(self.conv1(x), 0.2)
        #print(x.size())
        x = F.leaky_relu(self.batch1(self.conv2(x)), 0.2)
        #print(x.size())
        x = F.leaky_relu(self.batch2(self.conv3(x)), 0.2)
        #print(x.size())
        x = F.leaky_relu(self.batch3(self.conv4(x)), 0.2)
        #print(x.size())
        x = F.leaky_relu(self.batch3(self.conv5(x)), 0.2)
        #print(x.size())
        x = F.leaky_relu(self.batch2(self.conv6(x)), 0.2)
        x = self.conv7(x)
        #print(x.size())
        x = x.reshape(-1)
        #print(x.size())
        return x

class NetD_patch_high(nn.Module):
    def __init__(self):
        super(NetD_patch_high, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv7 = nn.Conv2d(256, 1, kernel_size=2, stride=2, padding=1, bias=False)

        self.batch1 = nn.BatchNorm2d(128, affine=False)
        self.batch2 = nn.BatchNorm2d(256, affine=False)
        self.batch3 = nn.BatchNorm2d(512, affine=False)

    def forward(self, x):
        #print(x.size())
        x = F.leaky_relu(self.conv1(x), 0.2)
        #print(x.size())
        x = F.leaky_relu(self.batch1(self.conv2(x)), 0.2)
        #print(x.size())
        x = F.leaky_relu(self.batch2(self.conv3(x)), 0.2)
        #print(x.size())
        x = F.leaky_relu(self.batch3(self.conv4(x)), 0.2)
        #print(x.size())
        x = F.leaky_relu(self.batch3(self.conv5(x)), 0.2)
        #print(x.size())
        x = F.leaky_relu(self.batch2(self.conv6(x)), 0.2)
        x = self.conv7(x)
        #print(x.size())
        x = x.reshape(-1)
        #print(x.size())
        return x


class NetG_srgan2(nn.Module):
    def __init__(self):
        super(NetG_srgan2, self).__init__()
        self.convInit = nn.Conv2d(3, 64, kernel_size=4, stride=4, padding=0, bias=False)
        self.convRes1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convRes2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convRes3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convRes4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convRes5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.convResInit = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)

        self.convOut = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)

        self.pix_shuf1 = nn.PixelShuffle(2)
        self.pix_shuf2 = nn.PixelShuffle(2)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        self.prelu5 = nn.PReLU()
        self.prelu6 = nn.PReLU()

        self.batch = nn.BatchNorm2d(64, affine=False)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.prelu1(self.convInit(x))
        xres1 = self.batch(self.convRes2(self.prelu2(self.batch(self.convRes1(x)))))+x
        xres2 = self.batch(self.convRes3(self.prelu3(self.batch(self.convRes3(xres1)))))+xres1
        xres3 = self.batch(self.convRes5(self.prelu4(self.batch(self.convRes4(xres2)))))+xres2
        x = self.batch(self.convResInit(xres3))+x
        # print(x.size())
        x = self.prelu5(self.pix_shuf1(self.conv1(x)))
        # print(x.size())
        x = self.prelu6(self.pix_shuf2(self.conv2(x)))
        # print(x.size())
        x = self.tanh(self.convOut(x))
        # print(x.size())
        return x


