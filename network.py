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
        self.convT4 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)

        self.conv_drop2d = nn.Dropout2d()

        self.batch1 = nn.BatchNorm2d(64, affine=False)
        self.batch2 = nn.BatchNorm2d(128, affine=False)
        self.batch3 = nn.BatchNorm2d(256, affine=False)
        self.batch4 = nn.BatchNorm2d(256, affine=False)
        self.batch5 = nn.BatchNorm2d(128, affine=False)
        self.batch6 = nn.BatchNorm2d(64, affine=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = F.relu(self.batch1(self.conv1(x)))
        x2 = F.relu(self.batch2(self.conv2(x1)))
        x2 = self.conv_drop2d(x2)
        x3 = F.relu(self.batch3(self.conv3(x2)))
        x4 = F.relu(self.conv4(x3))

        out = F.relu(self.batch4(self.convT1(x4)))
        out = F.relu(self.batch5(self.convT2(x3 + out)))
        out = self.conv_drop2d(out)
        out = F.relu(self.batch6(self.convT3(x2 + out)))
        out = self.tanh(self.convT4(out))
        return out


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
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=False)

        self.conv5bis = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv6 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=False)

        self.batch1 = nn.BatchNorm2d(128, affine=False)
        self.batch2 = nn.BatchNorm2d(256, affine=False)
        self.batch3 = nn.BatchNorm2d(512, affine=False)
        self.batch4 = nn.BatchNorm2d(512, affine=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.batch1(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.batch2(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.batch3(self.conv4(x)), 0.2)
        if self.mode == 'low':
            x = self.conv5(x)
        elif self.mode == 'high':
            x = F.leaky_relu(self.batch4(self.conv5bis(x)), 0.2)
            x = self.conv6(x)
        return x.squeeze()


class NetG_super(nn.Module):
    def __init__(self):
        super(NetG_super, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)

        self.convT1 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.convT2 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.convT3 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)
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
        out = F.relu(self.batch5(self.convT3(out + x1)))
        out = self.tanh(self.convT4(out))
        return out
