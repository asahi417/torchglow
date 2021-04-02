# Pytorch Glow
Pytorch implementation of [Glow](https://d4mucfpksywv.cloudfront.net/research-covers/glow/paper/glow.pdf).

### Setup
```
pip install git+https://github.com/asahi417/torchglow 
```

## Experiment
### Glow on CIFAR10
```shell script
torchglow-train-image -s 50000 --batch 256 --batch-init 512 --decay-lr --epoch-save 25 --epoch-valid 1 
```

### Glow on CelebA
```shell script
torchglow-train-image -d celeba -s 27000 --image-size 64 --batch 64 --batch-init 256 --n-level 6 --n-bits-x 5 --lr 0.0001 --additive-coupling --epoch-save 5 --epoch-valid 1 --debug
```

### Glow on Word Embedding
```shell script
torchglow-train-word -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 8 --unit-gaussian -m glove
torchglow-train-word -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 6 --unit-gaussian -m glove
torchglow-train-word -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 4 --unit-gaussian -m glove
torchglow-train-word -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 2 --unit-gaussian -m glove

torchglow-train-word -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 8 --unit-gaussian -m fasttext
torchglow-train-word -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 6 --unit-gaussian -m fasttext
torchglow-train-word -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 4 --unit-gaussian -m fasttext
torchglow-train-word -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 2 --unit-gaussian -m fasttext

torchglow-train-word -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 8 --unit-gaussian -m w2v
torchglow-train-word -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 6 --unit-gaussian -m w2v
torchglow-train-word -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 4 --unit-gaussian -m w2v
torchglow-train-word -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 2 --unit-gaussian -m w2v
```

## Evaluate
```shell script
torchglow-eval-word
```
