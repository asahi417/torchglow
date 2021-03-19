""" Solve analogy task by word embedding model """
import logging
import argparse
import os
from glob import glob
from itertools import chain

import pandas as pd
import torchglow

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
DATA = ['sat', 'u2', 'u4', 'google', 'bats']


def cos_similarity(a_, b_):
    inner = sum(list(map(lambda x: x[0] * x[1], zip(a_, b_))))
    norm_a = sum(list(map(lambda x: x * x, a_))) ** 0.5
    norm_b = sum(list(map(lambda x: x * x, b_))) ** 0.5
    if norm_b * norm_a == 0:
        return -100
    return inner / (norm_b * norm_a)


def diff(list_a, list_b):
    assert len(list_a) == len(list_b)
    return list(map(lambda x: x[0] - x[1], zip(list_a, list_b)))


def main(model_type: str):
    argument_parser = argparse.ArgumentParser(description='Model evaluation on analogy test.')
    argument_parser.add_argument('-b', '--batch', help='batch size', default=128, type=int)
    argument_parser.add_argument('--checkpoint-path', help='model checkpoint', default='./ckpt/{}/*'.format(model_type), type=str)
    argument_parser.add_argument('-o', '--output-dir', help='directory to export model weight file',
                                 default='./eval_output/{}/analogy_result.csv'.format(model_type), type=str)
    argument_parser.add_argument('--add-baseline', help='add baseline result', action='store_true')
    opt = argument_parser.parse_args()
    checkpoint_paths = glob(opt.checkpoint_path)
    result = []
    if model_type == 'fasttext':
        model_instance = torchglow.GlowFasttext
    elif model_type == 'bert':
        model_instance = torchglow.GlowBERT
    else:
        raise ValueError('unknown model type: {}'.format(model_type))
    base_prediction = torchglow.util.get_analogy_baseline()
    for checkpoint_path in checkpoint_paths:
        # all intermediate epoch
        epochs = [i.split('model.')[-1].replace('.pt', '') for i in glob('{}/model.*.pt'.format(checkpoint_path))]
        for e in epochs + [None]:
            if e is not None:
                model = model_instance(checkpoint_path=checkpoint_path, checkpoint_option={'epoch': e})
            else:
                model = model_instance(checkpoint_path=checkpoint_path)
                e = model.config.epoch_elapsed

            def get_word_pairs(word_pairs):
                all_pairs = list(chain(*[[o['stem']] + o['choice'] for o in word_pairs]))
                if model_type == 'fasttext':
                    if model.data_format == 'fasttext':
                        return list(set(list(chain(*all_pairs))))
                    elif model.data_format == 'relative':
                        all_pairs = [torchglow.util.word_pair_format(d) for d in all_pairs]
                        if model.vocab is not None:
                            return list(filter(lambda x: x in model.vocab, all_pairs))
                        return all_pairs
                    else:
                        raise ValueError('unknown data format: {}'.format(model.data_format))
                elif model_type == 'bert':
                    return all_pairs
                else:
                    raise ValueError('unknown model type: {}'.format(model_type))

            for i in DATA:
                tmp_result = model.parameter
                tmp_result['epoch'] = e
                tmp_result['data'] = i
                val, test = torchglow.util.get_analogy_dataset(i)
                # cache embedding
                data = get_word_pairs(val + test)
                vector = model.embed(data, batch=opt.batch)
                latent_dict = {str(k): v for k, v in zip(data, vector)}

                def get_prediction(single_data):
                    """ OOV should only happen in `fasttext` of `pair` format. """
                    if model_type == 'fasttext':
                        if model.data_format == 'fasttext':
                            diff_s = diff(latent_dict[single_data['stem'][0]], latent_dict[single_data['stem'][1]])
                            diff_c = [diff(latent_dict[a], latent_dict[b]) for a, b in single_data['choice']]
                            sim = [cos_similarity(diff_s, c) for c in diff_c]
                        elif model.data_format == 'relative':
                            stem = torchglow.util.word_pair_format(single_data['stem'])
                            choice = [torchglow.util.word_pair_format(d) for d in single_data['choice']]
                            if stem not in model.vocab:
                                return None
                            sim = [cos_similarity(latent_dict[stem], latent_dict[c]) if c in latent_dict else -100
                                   for c in choice]
                        else:
                            raise ValueError('unknown data format: {}'.format(model.data_format))
                    elif model_type == 'bert':
                        assert model.data_format == 'bert', model.data_format
                        v = latent_dict[str(single_data['stem'])]
                        v_c = [latent_dict[str(c)] for c in single_data['choice']]
                        sim = [cos_similarity(v, _v_c) for _v_c in v_c]
                    else:
                        raise ValueError('unknown model type: {}'.format(model_type))

                    pred = sim.index(max(sim))
                    if sim[pred] == -100:
                        return None
                    return pred

                for prefix, data in zip(['test', 'valid'], [test, val]):
                    prediction = [get_prediction(o) for o in data]
                    tmp_result['oov_{}'.format(prefix)] = len([p for p in prediction if p is None])
                    prediction = [p if p is not None else base_prediction[i][prefix][n] for n, p in enumerate(prediction)]
                    accuracy = [o['answer'] == p for o, p in zip(data, prediction)]
                    tmp_result['accuracy_{}'.format(prefix)] = sum(accuracy)/len(accuracy)
                tmp_result['accuracy'] = (tmp_result['accuracy_test'] * len(test) +
                                          tmp_result['accuracy_valid'] * len(val)) / (len(val) + len(test))
                result.append(tmp_result)

            del model

    # drop common config keys to keep only what different across models
    k = result[0].keys()
    k = [k_ for k_ in k if len(set([a_[k_] for a_ in result])) > 1]
    result = [{_k: r[_k] for _k in k} for r in result]

    if opt.add_baseline:
        # add fasttext baseline
        logging.info('fetch fasttext baseline')
        for baseline in ['fasttext_diff', 'concat_relative_fasttext', 'relative_init']:
            base_prediction = torchglow.util.get_analogy_baseline(baseline)
            for i in DATA:
                tmp_result = {k_: None for k_ in k}
                tmp_result['model_type'] = baseline
                tmp_result['data'] = i
                val, test = torchglow.util.get_analogy_dataset(i)
                for prefix, data in zip(['test', 'valid'], [test, val]):
                    prediction = base_prediction[i][prefix]
                    accuracy = [o['answer'] == p for o, p in zip(data, prediction)]
                    tmp_result['accuracy_{}'.format(prefix)] = sum(accuracy) / len(accuracy)
                tmp_result['accuracy'] = (tmp_result['accuracy_test'] * len(test) +
                                          tmp_result['accuracy_valid'] * len(val)) / (len(val) + len(test))
                result.append(tmp_result)

    df = pd.DataFrame(result)
    os.makedirs(os.path.basename(opt.output_dir), exist_ok=True)
    df.to_csv(opt.output_dir)


def main_bert():
    main('bert')


def main_fasttext():
    main('fasttext')

