# TorchGlow
Pytorch implementation of [Glow](https://d4mucfpksywv.cloudfront.net/research-covers/glow/paper/glow.pdf).

### Setup
```
pip install git+https://github.com/asahi417/torchglow 
```

## Model Training
### CIFAR10
```shell script
torchglow-train-image -s 50000 --batch 256 --batch-init 512 --decay-lr --epoch-valid 1 
```

```shell script
torchglow-generate --checkpoint-path ./ckpt/image/cifar10 --all-epoch -s 64 -b 32 -n 1
torchglow-latent --checkpoint-path ./ckpt/image/cifar10 --all-epoch 
```

### CelebA
```shell script
torchglow-train-image -d celeba -s 27000 --image-size 64 --batch 64 --batch-init 256 --n-level 6 --n-bits-x 5 --lr 0.0001 --additive-coupling --epoch-save 5 --epoch-valid 1 --debug
```
