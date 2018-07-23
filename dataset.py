import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import os

class CelebADatasetSup(torch.utils.data.Dataset):
    def __init__(self, imgFolder, f_bruit,transform=transforms.ToTensor()):
        super(CelebADatasetSup, self).__init__()
        self.imgFolder = imgFolder
        self.transform = transform
        self.f_bruit = f_bruit
        self.list = os.listdir(self.imgFolder)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        file = os.path.join(self.imgFolder,self.list[i])

        image = Image.open(file).crop((15,15,175,175))
        imgx = self.transform(image)
        if imgx.size(0) == 1:
            imgx = imgx.expand(3, imgx.size(1), imgx.size(2))
        imgxb = self.f_bruit(imgx)
        imgx2 = F.upsample(imgxb.unsqueeze(0), scale_factor=2, mode='nearest').squeeze(0)
        imgxb2 = self.f_bruit(imgx2)
        return imgx, imgx2, imgxb, imgxb2


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, imgFolder,f_bruit,transform=transforms.ToTensor()):
        super(CelebADataset, self).__init__()
        self.imgFolder = imgFolder
        self.transform = transform
        self.list = os.listdir(self.imgFolder)
        self.f_bruit = f_bruit

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        file = os.path.join(self.imgFolder,self.list[i])
        image = Image.open(file).crop((15,15,175,175))
        imgx = self.transform(image)
        if imgx.size(0) == 1:
            imgx = imgx.expand(3, imgx.size(1), imgx.size(2))
        imgxb = self.f_bruit(imgx)
        return imgx, imgxb


class CelebADatasetComp(torch.utils.data.Dataset):

    def __init__(self, imgFolder, idFile, attrFile, transform=transforms.ToTensor()):
        super(CelebADatasetComp, self).__init__()
        self.imgFolder = imgFolder
        with open(idFile) as f:
            lines = np.array([l[:-1].split() for l in f.readlines()])
        self.ids = torch.LongTensor(np.array(lines[:, 1], int))
        self.files = np.array(lines[:, 0])
        with open(attrFile) as f:
            lines = np.array([l[:-1].split() for l in f.readlines()])
        self.attrs = torch.FloatTensor(np.array(lines[:,1:], int))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filex = os.path.join(self.imgFolder, self.files[i])
        imgx = self.transform(Image.open(filex))
        if(imgx.size(0) == 1):
            imgx = imgx.expand(3, imgx.size(1), imgx.size(2))
        idx = self.ids[i]
        attrx = self.attrs[i]
        return imgx, idx, attrx, filex


class CelebADatasetMultyFaces(torch.utils.data.Dataset):
    def __init__(self, imgFolder, f_bruit, transform=transforms.ToTensor()):
        super(CelebADatasetMultyFaces, self).__init__()
        self.imgFolder = imgFolder
        self.transform = transform
        self.list = os.listdir(self.imgFolder)
        self.f_bruit = f_bruit

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):

        file = os.path.join(self.imgFolder, self.list[i])
        list_path = os.listdir(file)
        list_img = []
        list_imgb = []
        for i in list_path:
            print("i = ", i)
            with open(os.path.join(file,str(i))) as f:
                image = Image.open(f).crop((15,15,175,175))
                image = self.transform(image)
                if image.size(0) == 1:
                    image = image.expand(3, image.size(1), image.size(2))
            list_img.append(image)
            list_imgb.append(self.f_bruit(image))
        return list_img, list_imgb
