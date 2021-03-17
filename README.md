# Pytorch Glow
Pytorch implementation of [Glow](https://d4mucfpksywv.cloudfront.net/research-covers/glow/paper/glow.pdf).

### Setup
```
pip install git+https://github.com/asahi417/torchglow 
```

## Experiment
### CIFAR10
```shell script
torchglow-train-image --lr 0.005 -e 2000 --epoch-valid 1 --num-workers 8 --epoch-warm 5 --optimizer adamax --debug --progress-interval 10
```
