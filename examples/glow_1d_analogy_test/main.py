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
    inner = sum(list(map(lambda x: x[0] * x[1], zip(a_, b_))))
    norm_a = sum(list(map(lambda x: x * x, a_))) ** 0.5
    norm_b = sum(list(map(lambda x: x * x, b_))) ** 0.5
    return inner / (norm_b * norm_a)


def get_options():
    parser = argparse.ArgumentParser(description='Train Glow model on built-in dataset.')
    # model parameter
    parser.add_argument('--checkpoint-path', help='model checkpoint', default='./ckpt/relative/*', type=str)
    parser.add_argument('-o', '--output-dir', help='directory to export model weight file',
                        default='./examples/glow_1d_analogy_test', type=str)
    parser.add_argument('-b', '--batch', help='batch size', default=128, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    checkpoint_paths = glob(opt.checkpoint_path)
    # checkpoint_paths = opt.checkpoint_path.split(',')
    result = []
    for checkpoint_path in checkpoint_paths:
        epochs = [i.split('model.')[-1].replace('.pt', '') for i in glob('{}/model.*.pt'.format(checkpoint_path))]
        for e in epochs + [None]:
            if e is not None:
                model = torchglow.GlowWordEmbedding(checkpoint_path=checkpoint_path, checkpoint_option={'epoch': e})
            else:
                model = torchglow.GlowWordEmbedding(checkpoint_path=checkpoint_path)

            for i in DATA:
                tmp_result = {'model_type': model.config.model_type, 'n_flow_step': model.config.n_flow_step, 'data': i,
                              'epoch': model.config.epoch_elapsed, 'unit_gaussian': model.config.unit_gaussian}
                val, test = get_dataset_raw(i)
                all_pairs = list(chain(*[[o['stem']] + o['choice'] for o in val + test]))
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
                    prediction = [p if p is not None else BASE_PREDICTION[i][prefix][n] for n, p in enumerate(prediction)]
                    accuracy = [o['answer'] == p for o, p in zip(data, prediction)]
                    tmp_result['accuracy_{}'.format(prefix)] = sum(accuracy)/len(accuracy)
                tmp_result['accuracy'] = (tmp_result['accuracy_test'] * len(test) +
                                          tmp_result['accuracy_valid'] * len(val)) / (len(val) + len(test))
                result.append(tmp_result)

    df = pd.DataFrame(result)
    print(df)
    df.to_csv('{}/result.csv'.format(opt.output_dir))
