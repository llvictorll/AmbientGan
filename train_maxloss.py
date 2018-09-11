import torch
import torch.optim as optim
from network import *
from module_bruit import F_bruit, Patch_block, Sup_res1, Sup_res2
import torchvision.transforms as transforms
from itertools import zip_longest
from utils import *

from sacred import Experiment

from tqdm import tqdm
from dataset import CelebADataset, YoutubeFacesDataset

ex = Experiment('test')

@ex.config
def conf():
    device = 'cuda:0'
    netG = NetG_srgan().to(device)
    netDlow = NetD_super().to(device)
    netDhigh = NetD_patch().to(device)
    optimizerG = optim.Adam(netG.parameters(), 0.0004, betas=(0.5, 0.999))
    optimizerDlow = optim.Adam(netDlow.parameters(), 0.0004, betas=(0.5, 0.999))
    optimizerDhigh = optim.Adam(netDhigh.parameters(), 0.0005, betas=(0.5, 0.999))
    f_bruit = Sup_res2
    epoch = 5
    cuda = True
    param = None
    f = f_bruit(param)

    datasetCeleb = CelebADataset("/net/girlschool/besnier/CelebA_dataset/img_align_celeba/",
                                   f,
                                   transforms.Compose([transforms.Resize(64),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                       ]))

    dataloaderCeleb = torch.utils.data.DataLoader(datasetCeleb, batch_size=64, shuffle=True, num_workers=1, drop_last=True)

    datasetYtrain = YoutubeFacesDataset("/net/girlschool/besnier/YoutubeFaces",
                                        f,
                                        0,
                                        80,
                                        transforms.Compose([transforms.Resize(64),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                            ]))

    trainloaderY = torch.utils.data.DataLoader(datasetYtrain, batch_size=64, shuffle=True, num_workers=1, drop_last=True)

    datasetYtest = YoutubeFacesDataset("/net/girlschool/besnier/YoutubeFaces",
                                       f,
                                       80,
                                       0,
                                       transforms.Compose([transforms.Resize(64),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                          ]))

    testloaderY = torch.utils.data.DataLoader(datasetYtest, batch_size=64, shuffle=True, num_workers=1, drop_last=True)


@ex.main
def main(netG, netDlow, netDhigh, f_bruit, epoch, device, param, cuda, dataloaderCeleb, trainloaderY, testloaderY, optimizerG, optimizerDlow, optimizerDhigh):
    file = 'AmbientGAN/train_patchY/max_loss_Y'
    print(file)
    sauvegarde_init(file)
    save_net(file, netG, netDlow, netDhigh)
    netG.train()
    netDlow.train()
    netDhigh.train()
    cpt = 0
    dTrue = []
    dFalse = []
    mse_train = []
    module_bruit = f_bruit(param).to(device)
    turn = True
    bar_epoch = tqdm(range(epoch))
    for _ in bar_epoch:
        for i, (xf, xbf), (xh, xbh) in zip_longest(tqdm(range(len(trainloaderY))), trainloaderY, dataloaderCeleb, fillvalue=(None, None)):
            if xf is None:
                break

            real_label = torch.FloatTensor(xf.size(0)).fill_(.9).to(device)
            fake_label = torch.FloatTensor(xf.size(0)).fill_(.1).to(device)
            real_label_patch = torch.FloatTensor(xf.size(0) * 4 * 4).fill_(.9).to(device)
            fake_label_patch = torch.FloatTensor(xf.size(0) * 4 * 4).fill_(.1).to(device)
            if cuda:
                xbf = xbf.cuda()
                xf = xf.cuda()
            ########################
            # train D
            #######################
            optimizerDlow.zero_grad()          # avec les images de femmes

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

            if xh is not None:                  # avec les images d'hommes
                xh = xh.cuda()
                xbh = xbh.cuda()
                # train Dhigh
                optimizerDhigh.zero_grad()
                # avec de vrais labels
                outputTrue2 = netDhigh(xh)
                lossDT2 = F.binary_cross_entropy_with_logits(outputTrue2, real_label_patch)

                # avec de faux labels
                fake = netG(xbh)
                outputFalse2 = netDhigh(fake)
                lossDF2 = F.binary_cross_entropy_with_logits(outputFalse2, fake_label_patch)
                (lossDF2 + lossDT2).backward()
                optimizerDhigh.step()

            ######################
            # train G
            ######################
            optimizerG.zero_grad()
            if xh is not None:
                xh = xh.cuda()
                xbh = xbh.cuda()
                outputGhigh = netG(xbh)
                outputDhigh1 = netDhigh(outputGhigh)
                lossmodule = F.binary_cross_entropy_with_logits(outputDhigh1, real_label_patch)
                losscomp = F.l1_loss(outputGhigh, xh)
            else:
                losscomp = 0
                lossmodule = 0

            outputG = netG(xbf)
            imagebruit1 = module_bruit(outputG, b=True)
            imagebruit2 = module_bruit(outputG, b=True)
            outputDhigh = netDhigh(outputG)
            #outputDlow = netDlow(imagebruit1)

            lossmax = F.l1_loss(imagebruit1, imagebruit2)
            losshigh = F.binary_cross_entropy_with_logits(outputDhigh, real_label_patch)
            #losslow = F.binary_cross_entropy_with_logits(outputDlow, real_label)

            (1*losshigh+100*losscomp+1*lossmodule-lossmax).backward()
            optimizerG.step()

            mse_train.append(F.mse_loss(outputG, xf).data.mean())
            bar_epoch.set_postfix({"Dataset": np.array(dTrue).mean(),
                                   "G": np.array(dFalse).mean(),
                                   "lossMSE": np.array(mse_train).mean()})

            #####################
            # test
            #####################
            if i % 250 == 0 and i > 1:
                testbar = tqdm(range(len(testloaderY)))
                mse_test = []
                ref = []
                for j, (xhqt, xlqt) in zip(testbar, testloaderY):
                    if j > 100:
                        break
                    if turn:
                        save_xb = xlqt
                        print_img(xhqt, 'image_de_base_sans_bruit', file)
                        print_img(F.upsample(xlqt, scale_factor=2), 'ref_upsampling', file)
                        turn = False
                        if cuda:
                            save_xb = save_xb.cuda()
                    if cuda:
                        xlqt = xlqt.cuda()
                        xhqt = xhqt.cuda()

                    output = netG(xlqt).detach()
                    mse_test.append(F.mse_loss(output, xhqt).data.mean())
                    ref.append(F.mse_loss(F.upsample(xlqt, scale_factor=2), xhqt).data.mean())

                    testbar.set_postfix({"qualité": np.array(mse_test).mean(),
                                         "ref": np.array(ref).mean()})
                sauvegarde(file, np.array(dTrue).mean(), np.array(dFalse).mean(),
                           np.array(mse_test).mean(), np.array(mse_train).mean(), np.array(ref).mean())

                printG(save_xb, cpt, netG, file)
                cpt += 1
                mse_train = []
                dTrue = []
                dFalse = []

        for g in optimizerDlow.param_groups:
            g['lr'] = g['lr']*0.9
        for g in optimizerDhigh.param_groups:
            g['lr'] = g['lr'] * 0.9
        for g in optimizerG.param_groups:
            g['lr'] = g['lr'] * 0.9


if __name__ == '__main__':
    ex.run()
