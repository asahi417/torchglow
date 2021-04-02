import logging
import torchglow
import torchvision
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


model = torchglow.Glow(checkpoint_path='ckpt')
# model.reconstruct()
# out = model.embed_data(4)
# print(len(out[0]))
# print(out)
model.generate(export_path='tmp1.png', eps_std=1.2)
model.generate(export_path='tmp2.png', eps_std=1.2)
model.generate(export_path='tmp3.png', eps_std=1.2)
