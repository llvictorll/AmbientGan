import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class F_bruit(nn.Module):
    def __init__(self, param):
        super(F_bruit, self).__init__()
        self.param = param

    def forward(self, x):
        self.r = torch.rand(x.size())
        self.r = np.where(self.r < self.param, 0, 1)
        if isinstance(x, torch.cuda.FloatTensor):
            self.r = torch.tensor(self.r, device='cuda', dtype=torch.float32, requires_grad=False)
        else:
            self.r = torch.tensor(self.r, device='cpu', dtype=torch.float32, requires_grad=False)

        return self.r * x


class Patch_block(nn.Module):
    def __init__(self, taille):
        super(Patch_block, self).__init__()
        self.taille = taille

    def forward(self, x):
        w = np.random.randint(0, 64 - self.taille)
        h = np.random.randint(0, 64 - self.taille)
        self.r = np.zeros(x.size())
        self.r[:, w:w + self.taille, h:h + self.taille] = 1
        if isinstance(x, torch.cuda.FloatTensor):
            self.r = torch.tensor(self.r, device='cuda', dtype=torch.float32, requires_grad=False)
        else:
            self.r = torch.tensor(self.r, device='cpu', dtype=torch.float32, requires_grad=False)
        return self.r * x


class Sup_res1(nn.Module):
    def __init__(self, param=None):
        super(Sup_res1, self).__init__()
        self.param = param

    def forward(self, x, b=False):

        copy_x = x.cpu()
        randi = torch.LongTensor(np.sort(random.sample(range(64), 32)))
        randj = torch.LongTensor(np.sort(random.sample(range(64), 32)))
        copy_x = torch.index_select(copy_x, -1, randi)
        copy_x = torch.index_select(copy_x, -2, randj)

        if b:
            return copy_x.cuda()

        return copy_x

class Sup_res2(nn.Module):
    def __init__(self, param=None):
        super(Sup_res2, self).__init__()
        self.param = param

    def forward(self, x, b=False):
        copy_x = x.cpu()
        i = []
        j = []
        for h in range(0, 64, 2):
            i.append(random.randint(h, h + 1))
            j.append(random.randint(h, h + 1))

        randi = torch.LongTensor(i)
        randj = torch.LongTensor(j)
        copy_x = torch.index_select(copy_x, -1, randi)
        copy_x = torch.index_select(copy_x, -2, randj)

        if b:
            return copy_x.cuda()

        return copy_x.squeeze()

class Sup_res3(nn.Module):
    def __init__(self, param=None):
        super(Sup_res3, self).__init__()
        self.param = param

    def forward(self, x, b=False):
        copy_x = x.cpu()
        i = []
        j = []
        for h in range(0, 64, 2):
            i.append(random.randint(h, h + 1))
            j.append(random.randint(h, h + 1))

        randi = torch.LongTensor(i)
        randj = torch.LongTensor(j)
        copy_x = torch.index_select(copy_x, -1, randi)
        copy_x = torch.index_select(copy_x, -2, randj)
        r = torch.randn_like(copy_x)*0.2
        copy_x += r

        if b:
            copy_x = torch.tensor(copy_x, device='cuda', requires_grad=False)
        else:
            copy_x = torch.tensor(copy_x, device='cpu', requires_grad=False)

        return copy_x.squeeze()


class ConvNoise(nn.Module):
    def __init__(self, conv_size, noise_variance):
        super().__init__()
        self.conv_size = conv_size
        self.noise_variance = noise_variance

    def forward(self, x, device='cpu'):
        x_measured = x.clone()
        noise = torch.randn_like(x_measured) * self.noise_variance
        noise = torch.tensor(noise, device=device, requires_grad=False)
        eps = torch.ones(1, 1, self.conv_size, self.conv_size, device=device) / (self.conv_size * self.conv_size)
        for i in range(3):
            x_measured[:, i:i + 1] = F.conv2d(x[:, i:i + 1], eps, stride=1, padding=self.conv_size//2)
        x_measured = x_measured + noise

        return x_measured.clamp(-0.999, 0.9999)
