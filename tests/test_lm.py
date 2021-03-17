import torchglow

if __name__ == '__main__':
    model = torchglow.lm.BERT('albert-base-v1', mode='relative')
    model.get_embedding(['Paris', 'France'])

    model = torchglow.lm.BERT('albert-base-v1', mode='cls')
    model.get_embedding(['Paris is the capital of France'])
