import torch
import torch.optim as optim
from network import *
from module_bruit import F_bruit, Patch_block, Sup_res1, Sup_res2
import torchvision.transforms as transforms
from itertools import zip_longest
from utils import *

from sacred import Experiment

from tqdm import tqdm
from dataset import CelebADataset

ex = Experiment('test')

@ex.config
def conf():
    device = 'cuda:0'
    netG = NetG_pix().to(device)
    netDlow = NetD_super('low').to(device)
    netDhigh = NetD_super('high').to(device)
    optimizerG = optim.Adam(netG.parameters(), 0.0002, betas=(0.5, 0.999))
    optimizerDlow = optim.Adam(netDlow.parameters(), 0.0001, betas=(0.5, 0.999))
    optimizerDhigh = optim.Adam(netDhigh.parameters(), 0.0001, betas=(0.5, 0.999))
    f_bruit = Sup_res2
    epoch = 15
    cuda = True
    param = None
    f = f_bruit(param)

    datasetH = CelebADataset("/net/girlschool/besnier/CelebA_dataset/multi_dataset/img_H",
                             f,
                             transforms.Compose([transforms.Resize(64),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ]))

    dataloaderH = torch.utils.data.DataLoader(datasetH, batch_size=64, shuffle=True, num_workers=1, drop_last=True)

    datasetF = CelebADataset("/net/girlschool/besnier/CelebA_dataset/multi_dataset/img_F",
                             f,
                             transforms.Compose([transforms.Resize(64),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ]))

    dataloaderF = torch.utils.data.DataLoader(datasetF, batch_size=64, shuffle=True, num_workers=1, drop_last=True)

@ex.main
def main(netG, netDlow, netDhigh, f_bruit, epoch, device, param, cuda, dataloaderH, dataloaderF, optimizerG, optimizerDlow, optimizerDhigh, l1, l2, l3):
    netG.train()
    netDlow.train()
    netDhigh.train()
    file = 'train_test_pixel_shuffle/'
    sauvegarde_init(file)
    cpt = 0
    mse = []
    ref = []
    dTrue = []
    dFalse = []
    module_bruit = f_bruit(param).to(device)
    turn = True
    bar_epoch = tqdm(range(epoch))
    for _ in bar_epoch:
        for i, (xf, xbf), (xh, xbh) in zip_longest(tqdm(range(len(dataloaderF))), dataloaderF, dataloaderH, fillvalue=(None, None)):
            if turn:
                print_img(xf,"référence_femme", file)
                save_xbf = xbf
                print_img(save_xbf, 'image_de_bruit_femme',file)
                turn = False
                if cuda:
                    save_xbf = save_xbf.cuda()

            real_label = torch.FloatTensor(xf.size(0)).fill_(.9).to(device)
            fake_label = torch.FloatTensor(xf.size(0)).fill_(.1).to(device)

            if cuda:
                xbf = xbf.cuda()
                xf = xf.cuda()
            ########################
            # train D
            #######################
            optimizerDlow.zero_grad()          # avec les images de f

            # avec de vrais labels
            outputTrue = netDlow(xbf)
            lossDT = F.binary_cross_entropy_with_logits(outputTrue, real_label)

            # avec de faux labels
            fake = module_bruit(netG(xbf), b=True)
            outputFalse = netDlow(fake)
            lossDF = F.binary_cross_entropy_with_logits(outputFalse, fake_label)
            (lossDF + lossDT).backward()
            optimizerDlow.step()

            dTrue.append(F.sigmoid(outputTrue).data.mean())
            dFalse.append(F.sigmoid(outputFalse).data.mean())
            if xh is not None:
                xh = xh.cuda()
                xbh = xbh.cuda()
                # train Dhigh
                optimizerDhigh.zero_grad()
                # avec de vrais labels
                outputTrue = netDhigh(xh)
                lossDT = F.binary_cross_entropy_with_logits(outputTrue, real_label)

                # avec de faux labels
                fake = netG(module_bruit(xh, b=True))
                outputFalse = netDhigh(fake)
                lossDF = F.binary_cross_entropy_with_logits(outputFalse, fake_label)
                (lossDF + lossDT).backward()
                optimizerDhigh.step()

            ######################
            # train G
            ######################
            optimizerG.zero_grad()

            # optim with high image
            if xbh is not None:
                xbh = xbh.cuda()
                outputGhigh = netG(xbh)
                outputDhigh = netDhigh(outputGhigh)
                losshigh = F.binary_cross_entropy_with_logits(outputDhigh, real_label)
            else:
                losshigh = 0
            # optim with low image
            outputG = netG(xbf)
            outputGlow = module_bruit(outputG, b=True)
            outputDlow = netDlow(outputGlow)
            losslow = F.binary_cross_entropy_with_logits(outputDlow, real_label)

            loss_sup = F.mse_loss(netG(outputGlow), outputG)

            (l1*losshigh + l2*losslow + l3 * loss_sup).backward()
            optimizerG.step()

            #####################
            # test
            #####################
            dTrue.append(F.sigmoid(outputTrue).data.mean())
            dFalse.append(F.sigmoid(outputFalse).data.mean())
            mse.append(F.mse_loss(netG(xbf).detach(), xf))
            ref.append(F.mse_loss(F.upsample(xbf, scale_factor=2), xf))
            dTrue.append(F.sigmoid(outputTrue).data.mean())
            dFalse.append(F.sigmoid(outputFalse).data.mean())

            bar_epoch.set_postfix({"Dataset": np.array(dTrue).mean(),
                                   "G": np.array(dFalse).mean(),
                                   "qualité": np.array(mse).mean(),
                                   "ref": np.array(ref).mean()})

            sauvegarde(file,np.array(dTrue).mean(), np.array(dFalse).mean(), np.array(mse).mean(), np.array(ref).mean())

            if i % 250 == 0:
                printG(save_xbf, cpt, netG, file)
                cpt += 1
                mse = []
                ref = []
                dTrue = []
                dFalse = []

            # truc = normal
            # tru2 = losslow * 0.01
            # truc3 = losslow * 0.1
            # truc4 = losslow * 10
            # truc5 = losslow * 0.1 and len(H)/2
            # truc6 = losslow * 0.1 and len(H)/4
            # truc7 = losslow * 0.1 and len(H)/8
            # truc8 = losslow * 0.1 and len(H)/16
            # truc9 = losshigh + 0.01*losslow + 10*loss_sup


if __name__ == '__main__':
    ex.run(config_updates={'l1': 1, 'l2': 0.1, 'l3': 1})
