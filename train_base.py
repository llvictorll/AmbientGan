import torch
import torch.optim as optim
from network import *
from module_bruit import F_bruit, Patch_block, Sup_res1, Sup_res2, Sup_res3, ConvNoise
import torchvision.transforms as transforms
from utils import *

from sacred import Experiment

from tqdm import tqdm
from dataset import CelebADataset, CelebADatasetSup, CrowdDataset

ex = Experiment('test')

@ex.config
def conf():
    device = 'cuda:0'
    netG = NetG().to(device)
    netD = NetD().to(device)
    optimizerG = optim.Adam(netG.parameters(), 0.0003, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), 0.0001, betas=(0.5, 0.999))
    epoch = 15
    param = None
    f = ConvNoise(17, 0.01)

    traindataset = CrowdDataset("/net/girlschool/besnier/Crow/train",
                                "/net/girlschool/besnier/Crow/indextrain.txt",
                                f,
                                transforms.Compose([transforms.ToPILImage(),
                                                    transforms.CenterCrop(1024),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                    ]),
                                transforms.Compose([transforms.ToTensor()
                                                   ]),
                                transforms.Compose([transforms.ToPILImage(),
                                                    transforms.CenterCrop(256),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                    ]))

    testdataset = CrowdDataset("/net/girlschool/besnier/Crow/test",
                               "/net/girlschool/besnier/Crow/indextest.txt",
                               f,
                               transforms.Compose([transforms.ToPILImage(),
                                                   transforms.CenterCrop(1024),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                   ]),
                                transforms.Compose([transforms.ToTensor()
                                                   ]),
                                transforms.Compose([transforms.ToPILImage(),
                                                    transforms.CenterCrop(256),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                    ]))

    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=8, shuffle=True, num_workers=1, drop_last=True)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=8, shuffle=True, num_workers=1, drop_last=True)


@ex.automain
def main(netG, netD, f, epoch, dataloader, optimizerG, optimizerD, device):
    file = 'AmbientGAN/base_ConVnoise'
    netG.train()
    netD.train()
    sauvegarde_init(file)
    cpt = 0
    dTrue = []
    dFalse = []
    module_bruit = f.to(device)
    bar_epoch = tqdm(range(epoch))
    turn = True
    for _ in bar_epoch:
        for i, (x, xb) in zip(tqdm(range(len(dataloader))), dataloader):
            if turn:
                print_img(x, "référence", file)
                print_img(xb, 'image_de_bruit', file)
                turn = False

            xb = xb.to(device)
            real_label = torch.FloatTensor(x.size(0)).fill_(.9).to(device)
            fake_label = torch.FloatTensor(x.size(0)).fill_(.1).to(device)
            noise = torch.randn(x.size(0), 256, 1, 1).to(device)

            # train D
            optimizerD.zero_grad()
            # avec de vrais labels
            outputTrue = netD(xb)
            lossDT = F.binary_cross_entropy_with_logits(outputTrue, real_label)

            # avec de faux labels
            fake = module_bruit(netG(noise), device=device)
            outputFalse = netD(fake)
            lossDF = F.binary_cross_entropy_with_logits(outputFalse, fake_label)
            (lossDF + lossDT).backward()
            optimizerD.step()

            # train G
            optimizerG.zero_grad()
            outputG = module_bruit(netG(noise), device=device)
            outputDbruit = netD(outputG)
            lossBruit = F.binary_cross_entropy_with_logits(outputDbruit, real_label)

            lossBruit.backward()
            optimizerG.step()

            # test
            dTrue.append(F.sigmoid(outputTrue).data.mean())
            dFalse.append(F.sigmoid(outputFalse).data.mean())

            bar_epoch.set_postfix({"Dataset": np.array(dTrue).mean(),
                                   "G": np.array(dFalse).mean()})

            if i % 250 == 0:
                printG(noise, cpt, netG, file)
                cpt += 1
                dTrue = []
                dFalse = []
