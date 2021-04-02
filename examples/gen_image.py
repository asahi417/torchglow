import logging
import torchglow
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


torchglow.util.fix_seed()

data = 'cifar10'

decoder = torchglow.get_image_decoder()
train, val = torchglow.get_dataset_image(data)
logging.info('initialize iterator')
train = iter(train)
val = iter(val)
logging.info('generating images')
for i in range(n_img):
    img = next(train)[0]
    decoder(img).save('./tests/img/test_data/{}.train.{}.png'.format(data, i))
    img = next(val)[0]
    decoder(img).save('./tests/img/test_data/{}.valid.{}.png'.format(data, i))
