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
    netG = NetG_srgan().to(device)
    netDlow = NetD_super('low').to(device)
    netDhigh = NetD_super('low').to(device)
    optimizerG = optim.Adam(netG.parameters(), 0.0002, betas=(0.5, 0.999))
    optimizerDlow = optim.Adam(netDlow.parameters(), 0.0001, betas=(0.5, 0.999))
    optimizerDhigh = optim.Adam(netDhigh.parameters(), 0.0001, betas=(0.5, 0.999))
    f_bruit = Sup_res2
    epoch = 15
    cuda = True
    param = None
    f = f_bruit(param)
    file = 'train_rec/'
    dataset = CelebADataset("/net/girlschool/besnier/CelebA_dataset/img_align_celeba/",
                             f,
                             transforms.Compose([transforms.Resize(64),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1, drop_last=True)


@ex.main
def main(netG, netDlow, netDhigh, f_bruit, epoch, device, param, cuda, dataloader, optimizerG, optimizerDlow, optimizerDhigh, l1, l2, file):
    netG.train()
    netDlow.train()
    netDhigh.train()
    sauvegarde_init(file)
    cpt = 0
    mse = []
    ref = []
    dTrue = []
    dFalse = []
    module_bruit = f_bruit(param).to(device)
    bar_epoch = tqdm(range(epoch))
    turn = True
    for _ in bar_epoch:
        turnn = True
        for i, (x, xb) in zip_longest(tqdm(range(len(dataloader))),  dataloader, fillvalue=(None, None)):
            if turn:
                print_img(x, "référence", file)
                save_xbf = xb
                print_img(xb, 'image_de_bruit', file)
                turn = False
                if cuda:
                    save_xbf = save_xbf.cuda()
            if turnn:
                turnn = False
                continue

            xbb = F.avg_pool2d(xb, 4, stride=2, padding=1)
            real_label = torch.FloatTensor(x.size(0)).fill_(.9).to(device)
            fake_label = torch.FloatTensor(x.size(0)).fill_(.1).to(device)

            if cuda:
                xbb = xbb.cuda()
                xb = xb.cuda()
                x = x.cuda()
            ########################
            # train D
            #######################
            optimizerDlow.zero_grad()                                                # avec les images de taille 16->32

            # avec de vrais labels
            outputTrue = netDlow(xb)
            lossDT = F.binary_cross_entropy_with_logits(outputTrue, real_label)

            # avec de faux labels
            fake = netG(xbb)
            outputFalse = netDlow(fake)
            lossDF = F.binary_cross_entropy_with_logits(outputFalse, fake_label)
            (lossDF + lossDT).backward()
            optimizerDlow.step()

            dTrue.append(F.sigmoid(outputTrue).data.mean())
            dFalse.append(F.sigmoid(outputFalse).data.mean())

            # train Dhigh
            optimizerDhigh.zero_grad()                                               # avec les images de taille 32->64
            # avec de vrais labels
            outputTrue = netDhigh(xb)
            lossDT = F.binary_cross_entropy_with_logits(outputTrue, real_label)
            # avec de faux labels
            fake = module_bruit(netG(xb), b=True)
            outputFalse = netDhigh(fake)
            lossDF = F.binary_cross_entropy_with_logits(outputFalse, fake_label)
            (lossDF + lossDT).backward()
            optimizerDhigh.step()

            ######################
            # train G
            ######################
            optimizerG.zero_grad()
            # optim with high image
            outputGhigh = module_bruit(netG(xb), b=True)
            outputDhigh = netDhigh(outputGhigh)
            losshigh = F.binary_cross_entropy_with_logits(outputDhigh, real_label)

            # optim with low image
            outputGlow = netG(xbb)
            outputDlow = netDlow(outputGlow)

            losslow = F.binary_cross_entropy_with_logits(outputDlow, real_label)

            (l1*losshigh + l2*losslow).backward()
            optimizerG.step()

            #####################
            # test
            #####################
            dTrue.append(F.sigmoid(outputTrue).data.mean())
            dFalse.append(F.sigmoid(outputFalse).data.mean())
            mse.append(F.mse_loss(netG(xb).detach(), x))
            ref.append(F.mse_loss(F.upsample(xb, scale_factor=2), x))
            dTrue.append(F.sigmoid(outputTrue).data.mean())
            dFalse.append(F.sigmoid(outputFalse).data.mean())

            bar_epoch.set_postfix({"Dataset": np.array(dTrue).mean(),
                                   "G": np.array(dFalse).mean(),
                                   "qualité": np.array(mse).mean(),
                                   "ref": np.array(ref).mean()})

            sauvegarde(file, np.array(dTrue).mean(), np.array(dFalse).mean(), np.array(mse).mean(), np.array(ref).mean())

            if i % 250 == 1:
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
    ex.run(config_updates={'l1': 1, 'l2': 2})
