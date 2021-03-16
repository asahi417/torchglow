""" Solve analogy task by word embedding model """
import os
import logging
import json
import argparse
from glob import glob
from itertools import chain

import pandas as pd
import torchglow

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# Analogy data
DATA = ['sat', 'u2', 'u4', 'google', 'bats']
FASTTEXT_PREDICTION = 'fasttext_prediction.json'
if not os.path.exists(FASTTEXT_PREDICTION):
    torchglow.util.open_compressed_file(
        url='https://raw.githubusercontent.com/asahi417/AnalogyDataset/master/fasttext_prediction.json', cache_dir='.')
with open(FASTTEXT_PREDICTION) as f:
    BASE_PREDICTION = json.load(f)


def get_dataset_raw(data_name: str):
    """ Get SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    cache_dir = './cache'
    root_url_analogy = 'https://github.com/asahi417/AnalogyDataset/releases/download/0.0.0'
    assert data_name in DATA, 'unknown data: {}'.format(data_name)
    if not os.path.exists('{}/{}'.format(cache_dir, data_name)):
        torchglow.util.open_compressed_file('{}/{}.zip'.format(root_url_analogy, data_name), cache_dir)
    with open('{}/{}/test.jsonl'.format(cache_dir, data_name), 'r') as f:
        test_set = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    with open('{}/{}/valid.jsonl'.format(cache_dir, data_name), 'r') as f:
        val_set = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    return val_set, test_set


def cos_similarity(a_, b_):
    inner = (a_ * b_).sum()
    norm_a = (a_ * a_).sum() ** 0.5
    norm_b = (b_ * b_).sum() ** 0.5
    return inner / (norm_b * norm_a)


def get_options():
    parser = argparse.ArgumentParser(description='Train Glow model on built-in dataset.')
    # model parameter
    parser.add_argument('--checkpoint-path', help='model checkpoint', required=True, type=str)
    parser.add_argument('-o', '--output-dir', help='directory to export model weight file', default='./ckpt', type=str)
    parser.add_argument('-b', '--batch', help='batch size', default=128, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    epochs = [i.split('model.')[-1].replace('.pt', '') for i in glob('{}/model.*.pt'.format(opt.checkpoint_path))]
    result = []
    for e in epochs + [None]:
        if e is not None:
            model = torchglow.GlowWordEmbedding(checkpoint_path=opt.checkpoint_path, checkpoint_option={'epoch': e})
        else:
            model = torchglow.GlowWordEmbedding(checkpoint_path=opt.checkpoint_path)

        for i in DATA:
            tmp_result = {'model_type': model.config.model_type, 'data': i, 'epoch': e, 'ckpt': opt.checkpoint_path}
            val, test = get_dataset_raw(i)
            all_pairs = list(chain(*[[[o['stem']] + o['choice']] for o in val + test]))
            print(all_pairs)
            all_pairs_format = ['__'.join(p).replace(' ', '_').lower() for p in all_pairs]
            all_pairs_format = list(filter(lambda x: x in model.vocab(), all_pairs_format))
            vector = model.embed(all_pairs_format, batch=opt.batch)
            latent_dict = {k: v for k, v in zip(all_pairs_format, vector)}

            def get_prediction(single_data):
                stem = '__'.join(single_data['stem']).replace(' ', '_').lower()
                choice = ['__'.join(c).replace(' ', '_').lower() for c in single_data['choice']]
                if stem not in model.vocab():
                    return None
                sim = [cos_similarity(latent_dict[stem], latent_dict[c]) if c in latent_dict else -100 for c in choice]
                pred = sim.index(max(sim))
                if sim[pred] == -100:
                    return None
                return pred


            for prefix, data in zip(['test', 'valid'], [test, val]):
                prediction = [get_prediction(o) for o in data]
                tmp_result['oov_{}'.format(prefix)] = len([p for p in prediction if p is None])
                prediction = [p if p is not None else BASE_PREDICTION[data][prefix][n] for n, p in enumerate(prediction)]
                accuracy = [o['answer'] == p for o, p in zip(data, prediction)]
                tmp_result['accuracy_{}'.format(prefix)] = sum(accuracy)/len(accuracy)
            tmp_result['accuracy'] = (tmp_result['accuracy_test'] * len(test) +
                                      tmp_result['accuracy_valid'] * len(val)) / (len(val) + len(test))
            result.append(tmp_result)
    print(pd.DataFrame(result))
    # pd.DataFrame(tmp_result).tocsv('./result.csv')
    # logging.info('finish evaluation: result was exported to {}'.format('./result.jsonl'))