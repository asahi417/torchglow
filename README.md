# Pytorch Glow
Pytorch implementation of [Glow](https://d4mucfpksywv.cloudfront.net/research-covers/glow/paper/glow.pdf).

### Setup
```
pip install git+https://github.com/asahi417/torchglow 
```

## Experiment
### CIFAR10
```shell script
torchglow-train-image --num-workers 8
```

### Fasttext
```shell script
torchglow-train-fasttext --num-workers 8 -e 30 --epoch-save 1 --filter-size 32 --lr 0.00001 \
  -s 950000 -b 16384 --batch-init 16384 --optimizer adamw --weight-decay 1e-6 \
  --n-flow 3 --unit-gaussian -m fasttext
```

### BERT
```shell script
torchglow-train-bert --num-workers 8 -e 30 --epoch-save 1 --lr 0.00001 \
  -s 950000 -b 1024 --batch-init 1024 --optimizer adamw --weight-decay 1e-6 \
  --n-flow 3 --unit-gaussian

```