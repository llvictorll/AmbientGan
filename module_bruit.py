import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class F_bruit(nn.Module):
    def __init__(self, param):
        super(F_bruit, self).__init__()
        self.param = param

    def forward(self, x):
        self.r = torch.rand(x.size())
        if isinstance(x, torch.cuda.FloatTensor):
            self.r = torch.cuda.FloatTensor(np.where(self.r < self.param, 0, 1))
        else:
            self.r = torch.FloatTensor(np.where(self.r < self.param, 0, 1))

        return self.r * x


class Patch_block(nn.Module):
    def __init__(self, taille):
        super(Patch_block, self).__init__()
        self.taille = taille

    def forward(self, x):
        w = np.random.randint(0, 64 - self.taille)
        h = np.random.randint(0, 64 - self.taille)
        self.r = np.ones(x.size())
        self.r[:, w:w + self.taille, h:h + self.taille] = 0
        if isinstance(x, torch.cuda.FloatTensor):
            self.r = torch.cuda.FloatTensor(self.r)
        else:
            self.r = torch.FloatTensor(self.r)
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
        r = torch.randn_like(copy_x)*0.1
        copy_x += r

        if b:
            return copy_x.cuda()

        return copy_x.squeeze()

