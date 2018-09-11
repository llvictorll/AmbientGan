import torch
import torch.optim as optim
from network import *
from module_bruit import F_bruit, Patch_block, Sup_res1, Sup_res2, Sup_res3, ConvNoise
import torchvision.transforms as transforms
from utils import *

from sacred import Experiment

from tqdm import tqdm
from dataset import CrowdDataset

ex = Experiment('test')

@ex.config
def conf():
    device = 'cuda:0'
    netG = NetG_srgan2().to(device)
    netD = NetD_patch().to(device)
    optimizerG = optim.Adam(netG.parameters(), 0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), 0.0004, betas=(0.5, 0.999))
    epoch = 100
    cuda = True
    file = '/SRGAN/Crow_SRGAN_face_replace2'
    f = ConvNoise(21, 0)

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
                               transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                   ]),
                                transforms.Compose([transforms.ToTensor()
                                                   ]),
                                transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                    ]))

    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=8, shuffle=True, num_workers=1, drop_last=True)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)


@ex.automain
def main(netG, netD, epoch, trainloader, testloader, optimizerG, optimizerD, file, device):
    print(file)
    netG.train()
    netD.train()
    sauvegarde_init(file)
    cpt = 0
    dTrue = []
    dFalse = []
    mse_train = []
    turn = 0
    bar_epoch = tqdm(range(epoch))
    for e in bar_epoch:
        for i, (x, xb, v, vb, (a,b,c,d)) in zip(tqdm(range(len(trainloader))), trainloader):
            v = v.to(device)
            vb = vb.to(device)

            # train D
            optimizerD.zero_grad()

            # avec de vrais labels
            outputTrue = netD(v)
            real_label = torch.FloatTensor(outputTrue.size()).fill_(.9).to(device)
            fake_label = torch.FloatTensor(outputTrue.size()).fill_(.1).to(device)
            lossDT = F.binary_cross_entropy_with_logits(outputTrue, real_label)

            # avec de faux labels
            fake = netG(vb)
            outputFalse = netD(fake)
            lossDF = F.binary_cross_entropy_with_logits(outputFalse, fake_label)
            (lossDF + lossDT).backward()
            optimizerD.step()

            # train G

            optimizerG.zero_grad()
            outputG = netG(vb)
            outputDG = netD(outputG)
            lossGAN = F.binary_cross_entropy_with_logits(outputDG, real_label)
            lossMSE = F.mse_loss(outputG, v)

            (0.01*lossGAN+lossMSE).backward()
            optimizerG.step()
            dTrue.append(F.sigmoid(outputTrue).data.mean())
            dFalse.append(F.sigmoid(outputFalse).data.mean())
            mse_train.append(lossMSE.data.mean())
            bar_epoch.set_postfix({"Dataset": np.array(dTrue).mean(),
                                   "G": np.array(dFalse).mean(),
                                   "lossMSE": np.array(mse_train).mean()})
            #test
            if i % 250 == 0 and i > 1:
                testbar = tqdm(range(len(testloader)))
                mse_test = []
                for j, (x, xb, v, vb, (a,b,c,d)) in zip(testbar, testloader):
                    v = v.to(device)
                    vb = vb.to(device)
                    x = x.to(device)
                    xb = xb.to(device)
                    if j > 800:
                        break
                    if turn < 10:
                        if e == 0 and i == 250:
                            print_img(x, 'image_de_base_sans_bruit', file+'/'+str(turn))
                            print_img(xb, 'ref', file+'/'+str(turn))
                        copy_xb = xb
                        outputt = netG(vb).detach()
                        copy_xb[:,:,b:b+outputt.size(2),a:a+outputt.size(3)] = outputt
                        print_img(copy_xb, 'g'+str(cpt), file+'/'+str(turn))
                        cpt += 1
                        turn += 1

                    output = netG(vb).detach()
                    mse_test.append(F.mse_loss(output, v[:,:,0:output.size(2),0:output.size(3)]).data.mean())

                    testbar.set_postfix({"qualité": np.array(mse_test).mean()})

                sauvegarde(file, np.array(mse_test).mean(), np.array(mse_train).mean())
                turn = 0

                mse_train = []
                dTrue = []
                dFalse = []

    for g in optimizerD.param_groups:
        g['lr'] = g['lr']*0.99
    for g in optimizerG.param_groups:
        g['lr'] = g['lr'] * 0.99
