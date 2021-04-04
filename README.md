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

## Glow on Word Embedding
As we observed that [pretrained GloVe model](https://nlp.stanford.edu/projects/glove/) works better in Analogy test,
here we use GloVe ([the largest model](http://nlp.stanford.edu/data/glove.840B.300d.zip) shared by the authors) as our
anchor model and test a few configuration of Glow model that is filter size from `[32, 64, 128, 256]` and number of flows from `[8, 10, 12]`.
