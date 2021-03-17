import torchglow

if __name__ == '__main__':
    # model = torchglow.lm.BERT('albert-base-v1', mode='mask')
    # model.get_embedding(['Paris', 'France'])
    #
    # model = torchglow.lm.BERT('albert-base-v1', mode='cls')
    # model.get_embedding(['Paris is the capital of France'])

    glow = torchglow.GlowBERT(lm_model='albert-base-v1', epoch=1, training_step=10, export_dir='tests')
    glow.train()
