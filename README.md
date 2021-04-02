# Pytorch Glow
Pytorch implementation of [Glow](https://d4mucfpksywv.cloudfront.net/research-covers/glow/paper/glow.pdf).

### Setup
```
pip install git+https://github.com/asahi417/torchglow 
```

## Experiment
### Glow on CIFAR10
```shell script
torchglow-train-image --num-workers 8 -s 50000
```

### Glow on Word Embedding
```shell script
torchglow-train-fasttext -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 8 --unit-gaussian -m glove
torchglow-train-fasttext -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 6 --unit-gaussian -m glove
torchglow-train-fasttext -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 4 --unit-gaussian -m glove
torchglow-train-fasttext -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 2 --unit-gaussian -m glove

torchglow-train-fasttext -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 8 --unit-gaussian -m fasttext
torchglow-train-fasttext -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 6 --unit-gaussian -m fasttext
torchglow-train-fasttext -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 4 --unit-gaussian -m fasttext
torchglow-train-fasttext -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 2 --unit-gaussian -m fasttext

torchglow-train-fasttext -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 8 --unit-gaussian -m w2v
torchglow-train-fasttext -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 6 --unit-gaussian -m w2v
torchglow-train-fasttext -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 4 --unit-gaussian -m w2v
torchglow-train-fasttext -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 262144 --batch-init 262144 --n-flow 2 --unit-gaussian -m w2v
```

### BERT
```shell script
torchglow-train-bert -e 25 --epoch-save 5 --lr 0.0001 -b 1024 --batch-init 1024 --n-flow 8 --unit-gaussian
torchglow-train-bert -e 25 --epoch-save 5 --lr 0.0001 -b 1024 --batch-init 1024 --n-flow 6 --unit-gaussian
torchglow-train-bert -e 25 --epoch-save 5 --lr 0.0001 -b 1024 --batch-init 1024 --n-flow 4 --unit-gaussian 
torchglow-train-bert -e 25 --epoch-save 5 --lr 0.0001 -b 1024 --batch-init 1024 --n-flow 2 --unit-gaussian
torchglow-train-bert -e 25 --epoch-save 5 --lr 0.0001 -b 1024 --batch-init 1024 --n-flow 8 --unit-gaussian --lm-model bert-large-cased
torchglow-train-bert -e 25 --epoch-save 5 --lr 0.0001 -b 1024 --batch-init 1024 --n-flow 6 --unit-gaussian --lm-model bert-large-cased
torchglow-train-bert -e 25 --epoch-save 5 --lr 0.0001 -b 1024 --batch-init 1024 --n-flow 4 --unit-gaussian --lm-model bert-large-cased
torchglow-train-bert -e 25 --epoch-save 5 --lr 0.0001 -b 1024 --batch-init 1024 --n-flow 2 --unit-gaussian --lm-model bert-large-cased
```


## Evaluate
```shell script
torchglow-eval-fasttext
torchglow-eval-bert
```
