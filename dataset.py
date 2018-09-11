import torchvision.transforms as transforms
from PIL import Image
import os
from math import *
from tqdm import tqdm
from module_bruit import *
import cv2
import utils
import torchvision.transforms as transforms
import torchvision.utils as vutils

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
        try:
            file = os.path.join(self.imgFolder,self.list[i])
            image = Image.open(file).crop((15,15,175,175))
            imgx = self.transform(image)
            if imgx.size(0) == 1:
                imgx = imgx.expand(3, imgx.size(1), imgx.size(2))
            imgxb = self.f_bruit(imgx.unsqueeze(0)).squeeze()

        except:
            imgx = torch.randn(1,3,64,64)
            imgxb = self.f_bruit(imgx).squeeze()
            return imgx, imgxb

        return imgx, imgxb

class CrowdDataset(torch.utils.data.Dataset):
    def __init__(self, imgFolder, index, f_bruit, transform=transforms.ToTensor(), transform2=transforms.ToTensor(), transform3=transforms.ToTensor()):
        self.imgFolder = imgFolder
        self.index = index
        with open(self.index) as f:
            lines = np.array([l[:-1].split() for l in f.readlines()])
        self.files = np.array(lines)
        self.transform = transform
        self.transform2 = transform2
        self.transform3 = transform3
        self.list = os.listdir(self.imgFolder)
        self.f_bruit = f_bruit

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            filex = os.path.join(self.imgFolder, self.files[i,0]+'_'+self.files[i,1]+".jpg")
            image = Image.open(filex)
            x, y, w, h = self.files[i, 2:6]
            imgx = self.transform2(image)
            if imgx.size(0) == 1:
                imgx = imgx.expand(1, 3, imgx.size(1), imgx.size(2))
            visage = imgx[:,int(y):int(y) + int(h), int(x):int(x) + int(w)]
            visage2 = self.f_bruit(visage.unsqueeze(0), 'cpu').squeeze()
            imgxb = imgx.clone()
            imgxb[:,int(y):int(y) + int(h), int(x):int(x) + int(w)] = visage2
            return self.transform(imgx), self.transform(imgxb), self.transform3(visage), self.transform3(visage2), [
                int(x), int(y), int(w), int(h)]
        except:
            x, y, w, h = self.files[i, 2:6]
            imgx = torch.randn(3,1024,1024)
            imgxb = imgx.clone()
            visage = torch.randn(3,128,128)
            visage2 = visage.clone()

            return self.transform(imgx), self.transform(imgxb), self.transform3(visage), self.transform3(visage2), [int(x),int(y),int(w),int(h)]

class CrowdDatasetspe(torch.utils.data.Dataset):
    def __init__(self, imgFolder, index, f_bruit, transform=transforms.ToTensor(), transform2=transforms.ToTensor(), transform3=transforms.ToTensor()):
        self.imgFolder = imgFolder
        self.index = index
        with open(self.index) as f:
            lines = np.array([l[:-1].split() for l in f.readlines()])
        self.files = np.array(lines)
        self.transform = transform
        self.transform2 = transform2
        self.transform3 = transform3
        self.list = os.listdir(self.imgFolder)
        self.f_bruit = f_bruit

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filex = os.path.join(self.imgFolder, self.files[i,0]+'_'+self.files[i,1]+".jpg")
        image = Image.open(filex)
        x, y, w, h = self.files[i, 2:6]
        imgx = self.transform2(image)
        if imgx.size(0) == 1:
            imgx = imgx.expand(1, 3, imgx.size(1), imgx.size(2))
        visage = imgx[:,int(y):int(y) + int(h), int(x):int(x) + int(w)]
        visage2 = self.f_bruit(visage)
        imgxb = imgx.clone()
        imgxb[:,int(y):int(y) + int(h), int(x):int(x) + int(w)] = visage2
        return self.transform(imgx), self.transform(imgxb), self.transform3(visage), self.transform3(visage2),[int(x),int(y),int(w),int(h)]


class CelebADatasetBase(torch.utils.data.Dataset):
    def __init__(self, imgFolder,transform=transforms.ToTensor()):
        super(CelebADatasetBase, self).__init__()
        self.imgFolder = imgFolder
        self.transform = transform
        self.list = os.listdir(self.imgFolder)
        l = self.list.copy()
        for f in l:
            if f[0] == '.':
                self.list.remove(f)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        filex = os.path.join(self.imgFolder, self.list[i])
        imgx = self.transform(Image.open(filex))
        if imgx.size(0) == 1:
            imgx = imgx.expand(3, imgx.size(1), imgx.size(2))
        return imgx, filex


class CelebADatasetComp(torch.utils.data.Dataset):

    def __init__(self, imgFolder, idFile, attrFile, transform=transforms.ToTensor()):
        super(CelebADatasetComp, self).__init__()
        self.imgFolder = imgFolder
        with open(idFile) as f:
            lines = np.array([l[:-1].split() for l in f.readlines() ])
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


class YoutubeFacesDataset(torch.utils.data.Dataset):
    def __init__(self, ImgFolder, f_bruit, inf, sup, transform=transforms.ToTensor()):
        super(YoutubeFacesDataset, self).__init__()
        self.ImgFolder = ImgFolder
        self.list_p = os.listdir(self.ImgFolder)  # Contient la liste des personnalités
        self.transform = transform
        self.f_bruit = f_bruit
        self.dir = []
        for k in tqdm(range(len(self.list_p))):
            L = os.path.join(self.ImgFolder, self.list_p[k])
            Llist = os.listdir(L)
            for j in range(len(Llist)):
                L2 = os.path.join(L, Llist[j])
                L2list = os.listdir(L2)
                for k in range(len(L2list)):
                    self.dir.append(os.path.join(L2, L2list[k]))
        if inf < sup:
            self.dir_final = self.dir[0:ceil(len(self.dir) * sup / 100)]
        else:
            self.dir_final = self.dir[ceil(len(self.dir) * inf / 100):]

    def __len__(self):
        return len(self.dir_final)

    def __getitem__(self, i):
        img = Image.open(self.dir[i - 1]).crop((5, 5, 250, 250))
        imgx = self.transform(img)
        if imgx.size(0) == 1:
            imgx = imgx.expand(3, imgx.size(1), imgx.size(2))

        imgxr = self.f_bruit(imgx)
        return imgx, imgxr

class CelebADatasetexo(torch.utils.data.Dataset):
    def __init__(self, imgFolder,f_bruit,transform=transforms.ToTensor()):
        super(CelebADatasetexo, self).__init__()
        self.imgFolder = imgFolder
        self.transform = transform
        self.list = os.listdir(self.imgFolder)
        self.f_bruit = f_bruit

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        try:
            file = os.path.join(self.imgFolder,self.list[i])
            image = Image.open(file).crop((15,15,175,175))
            imgx = self.transform(image)
            if imgx.size(0) == 1:
                imgx = imgx.expand(3, imgx.size(1), imgx.size(2))
            imgxb = self.f_bruit(imgx)
            imgxu = F_bruit(F.upsample(imgxb, scale_factor=2))

        except:
            imgx = torch.randn(3,64,64)
            imgxb = self.f_bruit(imgx)
            pix = F_bruit(0.5)
            imgxu = pix(F.upsample(imgxb, scale_factor=2))
            return imgx, imgxb, imgxu
        return imgx, imgxb, imgxu

class CrowdDataset2(torch.utils.data.Dataset):
    def __init__(self, imgFolder, f_bruit, transform=transforms.ToTensor()):
        self.imgFolder = imgFolder
        self.transform = transform
        self.list = os.listdir(self.imgFolder)
        self.f_bruit = f_bruit
        self.face_reco = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        file = os.path.join(self.imgFolder, self.list[i])
        image = Image.open(file).crop((15, 15, 175, 175))
        imgx = self.transform(image)
        if imgx.size(0) == 1:
            imgx = imgx.expand(3, imgx.size(1), imgx.size(2))
        imgxb = utils.torch2PIL(imgx)
        faces = self.face_reco.detectMultiScale(imgxb, 1.3, 5)
        plt.imshow(imgxb)
        plt.show()
        for (x, y, w, h) in faces:
            print("OKOKOKOKOKOKOKO")
            cv2.rectangle(imgxb, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = imgxb[y:y + h, x:x + w]
            imgxb[y:y + h, x:x + w] = self.f_bruit(self.transform(f), 'cpu')
            plt.imshow(imgxb)
            plt.show()

        return imgx, imgxb






