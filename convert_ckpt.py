import torchglow
from glob import glob

ckpts = glob('./ckpt/image/cifar10/model*')
for i in ckpts:
    epoch = int(i.split('model.')[-1].replace('.pt', ''))
    if epoch < 1300:
        continue
    print('loading from {}'.format(epoch))
    try:
        torchglow.Glow(checkpoint_path='./ckpt/image/cifar10', checkpoint_epoch=epoch)
    except ValueError:
        print('DONE')
