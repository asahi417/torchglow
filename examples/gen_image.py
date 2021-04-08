import logging
import torchglow

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
model = torchglow.Glow(checkpoint_path='ckpt/celeba_128')
# out = model.embed_data(4)
# print(len(out[0]))
# print(out)
model.generate(export_path='tmp1.png', eps_std=2.2)
model.generate(export_path='tmp2.png', eps_std=1.8)
model.generate(export_path='tmp3.png', eps_std=1.4)
model.generate(export_path='tmp4.png', eps_std=1.0)
model.generate(export_path='tmp5.png', eps_std=0.6)
model.generate(export_path='tmp6.png', eps_std=0.2)
