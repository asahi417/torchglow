# Pytorch Glow
Pytorch implementation of [Glow](https://d4mucfpksywv.cloudfront.net/research-covers/glow/paper/glow.pdf).

### Setup
```
pip install git+https://github.com/asahi417/torchglow 
```

## Experiment
### CIFAR10
```shell script
torchglow-train-image --num-workers 8 -s 50000
```

### Fasttext
```shell script
torchglow-train-fasttext --num-workers 8 -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 65536 --batch-init 65536 --n-flow 8 --unit-gaussian -m fasttext
torchglow-train-fasttext --num-workers 8 -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 65536 --batch-init 65536 --n-flow 6 --unit-gaussian -m fasttext
torchglow-train-fasttext --num-workers 8 -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 65536 --batch-init 65536 --n-flow 4 --unit-gaussian -m fasttext
torchglow-train-fasttext --num-workers 8 -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 65536 --batch-init 65536 --n-flow 2 --unit-gaussian -m fasttext

torchglow-train-fasttext --num-workers 8 -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 131072 --batch-init 131072 --n-flow 8 --unit-gaussian -m fasttext_diff
torchglow-train-fasttext --num-workers 8 -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 131072 --batch-init 131072 --n-flow 6 --unit-gaussian -m fasttext_diff
torchglow-train-fasttext --num-workers 8 -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 131072 --batch-init 131072 --n-flow 4 --unit-gaussian -m fasttext_diff
torchglow-train-fasttext --num-workers 8 -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 131072 --batch-init 131072 --n-flow 2 --unit-gaussian -m fasttext_diff

torchglow-train-fasttext --num-workers 8 -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 131072 --batch-init 131072 --n-flow 8 --unit-gaussian -m concat_relative_fasttext
torchglow-train-fasttext --num-workers 8 -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 131072 --batch-init 131072 --n-flow 6 --unit-gaussian -m concat_relative_fasttext
torchglow-train-fasttext --num-workers 8 -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 131072 --batch-init 131072 --n-flow 4 --unit-gaussian -m concat_relative_fasttext
torchglow-train-fasttext --num-workers 8 -e 45 --epoch-save 5 --filter-size 32 --lr 0.0001 -b 131072 --batch-init 131072 --n-flow 2 --unit-gaussian -m concat_relative_fasttext
```

### BERT
```shell script
torchglow-train-bert --num-workers 8 -e 25 --epoch-save 5 --lr 0.0001 -b 2048 --batch-init 2048 --n-flow 8 --unit-gaussian
torchglow-train-bert --num-workers 8 -e 25 --epoch-save 5 --lr 0.0001 -b 2048 --batch-init 2048 --n-flow 6 --unit-gaussian
torchglow-train-bert --num-workers 8 -e 25 --epoch-save 5 --lr 0.0001 -b 2048 --batch-init 2048 --n-flow 4 --unit-gaussian
torchglow-train-bert --num-workers 8 -e 25 --epoch-save 5 --lr 0.0001 -b 2048 --batch-init 2048 --n-flow 2 --unit-gaussian
```

