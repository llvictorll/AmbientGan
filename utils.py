import scipy.misc
import torchvision.utils as vutils
import numpy as np


def imshow(img):
    img = img / 2 + 0.5
    npimg = (img.cpu()).numpy()
    return np.transpose(npimg, (1, 2, 0))


def printG(x, k, netG):
    o = netG(x)
    scipy.misc.imsave('/net/girlschool/besnier/truc/g{}.png'.format(k), imshow(vutils.make_grid(o.data)))
    scipy.misc.imsave('current_faces.png', imshow(vutils.make_grid(o.data)))


def print_img(x, name):
    scipy.misc.imsave('/net/girlschool/besnier/truc/' + name + '.png', imshow(vutils.make_grid(x).data))


def sauvegarde_init():
    with open("/net/girlschool/besnier/truc/res.csv", 'a') as f:
        f.write('dTrue' + '\t' + 'dFalse' + '\t' + 'qualité' + '\t' + 'référence' + '\n')


def sauvegarde(*agr):
    with open("/net/girlschool/besnier/truc/res.csv", 'a') as f:
        for a in agr:
            f.write(str(a) + '\t')
        f.write('\n')
