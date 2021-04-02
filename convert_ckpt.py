import torchglow
from glob import glob

ckpts = glob('./ckpt/image/cifar10/model*')
for i in ckpts:
    epoch = int(i.split('model.')[-1].replace('.pt', ''))
    print('loading from {}'.format(epoch))
    try:
        torchglow.Glow(checkpoint_path='./ckpt/image/cifar', checkpoint_epoch=epoch)
    except ValueError:
        print('DONE')
