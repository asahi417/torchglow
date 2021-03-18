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
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 2 -m fasttext
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 4 -m fasttext
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 6 -m fasttext
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 8 -m fasttext
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 2 -m fasttext_diff
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 4 -m fasttext_diff
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 6 -m fasttext_diff
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 8 -m fasttext_diff
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 2 -m concat_relative_fasttext
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 4 -m concat_relative_fasttext
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 6 -m concat_relative_fasttext
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 8 -m concat_relative_fasttext

torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 2 --unit-gaussian -m fasttext
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 4 --unit-gaussian -m fasttext
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 6 --unit-gaussian -m fasttext
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 8 --unit-gaussian -m fasttext
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 2 --unit-gaussian -m fasttext_diff
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 4 --unit-gaussian -m fasttext_diff
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 6 --unit-gaussian -m fasttext_diff
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 8 --unit-gaussian -m fasttext_diff
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 2 --unit-gaussian -m concat_relative_fasttext
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 4 --unit-gaussian -m concat_relative_fasttext
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 6 --unit-gaussian -m concat_relative_fasttext
torchglow-train-fasttext --num-workers 8 -e 15 --epoch-save 3 --filter-size 32 --lr 0.00001 -s 950000 -b 32768 --batch-init 32768 --n-flow 8 --unit-gaussian -m concat_relative_fasttext
```

### BERT
```shell script
torchglow-train-bert --num-workers 8 -e 15 --epoch-save 2 --lr 0.00001 \
  -s 950000 -b 1024 --batch-init 1024 --optimizer adamw --weight-decay 1e-6 \
  --n-flow 3 --unit-gaussian
```