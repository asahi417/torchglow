""" Solve analogy task by word embedding model
TODO: add option to apply random/false prediction for OOV entry
"""
import logging
import argparse
import os
import json
from glob import glob
from itertools import chain
from copy import deepcopy
import pandas as pd

from gensim.models import KeyedVectors
from gensim.test.utils import datapath

import torchglow

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
torchglow.util.fix_seed(0)


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


def main():
    argument_parser = argparse.ArgumentParser(description='Model evaluation on analogy test.')
    argument_parser.add_argument('-b', '--batch', help='batch size', default=262144, type=int)
    argument_parser.add_argument('--checkpoint-path', help='model checkpoint', default='./ckpt/glow_word_embedding/*',
                                 type=str)
    argument_parser.add_argument('-o', '--output-file', help='directory to export model weight file',
                                 default=None, type=str)
    argument_parser.add_argument('--multi-choice', help='SAT style multi choice analogy test', action='store_true')
    opt = argument_parser.parse_args()

    if opt.multi_choice:
        main_mc(opt)
    else:
        main_flat(opt)


def main_flat(opt):
    checkpoint_paths = glob(opt.checkpoint_path)
    result = []
    for n, checkpoint_path in enumerate(checkpoint_paths):
        logging.info('checkpoint: {}/{}'.format(n, len(checkpoint_paths)))
        # all intermediate epoch
        epochs = sorted(
            [int(i.split('model.')[-1].replace('.pt', '')) for i in glob('{}/model.*.pt'.format(checkpoint_path))])
        for e in epochs:
            logging.info('\t epoch: {}/{}'.format(e, epochs))
            with open('{}/config.json'.format(checkpoint_path), 'r') as f:
                parameter = json.load(f)
            tmp_result = deepcopy(parameter)
            tmp_result['epoch'] = e
            tmp_result['analogy_data'] = 'google_analogy_test'
            model_path = '{}/gensim_model.{}.bin'.format(checkpoint_path, e)
            if not os.path.exists(model_path):
                model = torchglow.GlowWordEmbedding(checkpoint_path=checkpoint_path, checkpoint_epoch=e)
                if model.word_pair_input:
                    logging.info('\t * skip as is trained on word pair')
                    continue
                model.export_gensim_model(model_path, batch=opt.batch)
            model = KeyedVectors.load_word2vec_format(model_path, binary=True)
            out = model.evaluate_word_analogies(datapath('questions-words.txt'))
            tmp_result['accuracy'] = out[0]
            logging.info('\t * accuracy: {}'.format(out[0]))
            del model
            result.append(tmp_result)

    df = pd.DataFrame(result)
    if opt.output_file is None:
        output_file = './eval/analogy/accuracy.glow_word_embedding.gat.csv'
    else:
        output_file = opt.output_file
    if os.path.exists(output_file):
        tmp = pd.read_csv(output_file, index_col=0)
        df = pd.concat([tmp, df])
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file)
    logging.info('result file exported to {}'.format(output_file))


def main_mc(opt):
    checkpoint_paths = glob(opt.checkpoint_path)

    def get_word_pairs(word_pairs, keep_pair: bool = False):
        all_pairs = list(chain(*[[o['stem']] + o['choice'] for o in word_pairs]))
        if keep_pair:
            return all_pairs
        return list(set(list(chain(*all_pairs))))

    # whole word in analogy dataset
    whole_data = []
    all_sat_data = ['sat', 'u2', 'u4', 'google', 'bats']
    for i in all_sat_data:
        val, test = torchglow.util.get_analogy_dataset(i)
        whole_data += val + test

    logging.info('** cache prediction **')
    result = []
    for n, checkpoint_path in enumerate(checkpoint_paths):
        logging.info('checkpoint: {}/{}'.format(n, len(checkpoint_paths)))
        # all intermediate epoch
        epochs = sorted([int(i.split('model.')[-1].replace('.pt', '')) for i in glob('{}/model.*.pt'.format(checkpoint_path))])
        for e in epochs:
            logging.info('\t epoch: {}/{}'.format(e, epochs))
            with open('{}/config.json'.format(checkpoint_path), 'r') as f:
                parameter = json.load(f)
            logging.info('\t * cache embedding for all words')
            path = '{}/analogy_cache.{}.json'.format(checkpoint_path, e)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    tmp_ = json.load(f)
                    latent_dict_normalized = tmp_['norm']
                    latent_dict_original = tmp_['org']
            else:
                model = torchglow.GlowWordEmbedding(checkpoint_path=checkpoint_path, checkpoint_epoch=e)
                whole_word = get_word_pairs(whole_data, model.word_pair_input)
                if model.word_pair_input and model.vocab is not None:
                    whole_word = list(filter(lambda x: x[0] in model.vocab and x[1] in model.vocab, whole_word))
                elif model.vocab is not None:
                    whole_word = list(filter(lambda x: x in model.vocab, whole_word))
                vector, vector_original = model.embed(whole_word, batch=opt.batch, return_original_embedding=True)
                latent_dict_normalized = {str(k): v for k, v in zip(whole_word, vector)}
                latent_dict_original = {str(k): v for k, v in zip(whole_word, vector_original)}
                with open(path, 'w') as f:
                    json.dump({'norm': latent_dict_normalized, 'org': latent_dict_original}, f)
                del model

            for i in all_sat_data:
                tmp_result = deepcopy(parameter)
                tmp_result['epoch'] = e
                tmp_result['analogy_data'] = i
                word_pair_input = True if tmp_result['data'] == 'common_word_pairs' else False

                # cache embedding
                val, test = torchglow.util.get_analogy_dataset(i)
                logging.info('\t * compute accuracy')

                def get_prediction(single_data, latent_dict):
                    try:
                        if word_pair_input:
                            diff_s = latent_dict[str(single_data['stem'])]
                            diff_c = [latent_dict[str(c)] for c in single_data['choice']]
                        else:
                            diff_s = diff(latent_dict[single_data['stem'][0]], latent_dict[single_data['stem'][1]])
                            diff_c = [diff(latent_dict[a], latent_dict[b]) for a, b in single_data['choice']]
                        sim = [cos_similarity(diff_s, c) for c in diff_c]
                    except KeyError:
                        sim = [-100] * len(single_data['choice'])
                    pred = sim.index(max(sim))
                    if sim[pred] == -100:
                        return None
                    return pred

                for prefix, data in zip(['test', 'valid'], [test, val]):
                    prediction_norm = [get_prediction(o, latent_dict_normalized) for o in data]
                    prediction_org = [get_prediction(o, latent_dict_original) for o in data]
                    tmp_result['oov_{}'.format(prefix)] = len([p for p in prediction_norm if p is None])
                    tmp_result['pred_norm_{}'.format(prefix)] = {n: o['answer'] == p for n, (o, p) in
                                                                 enumerate(zip(data, prediction_norm)) if p is not None}
                    tmp_result['pred_org_{}'.format(prefix)] = {n: o['answer'] == p for n, (o, p) in
                                                                enumerate(zip(data, prediction_org)) if p is not None}
                result.append(tmp_result)

    logging.info('** aggregate accuracy **')
    for i in all_sat_data:
        val, test = torchglow.util.get_analogy_dataset(i)
        tmp_results = list(filter(lambda x: x['analogy_data'] == i, result))
        d = [list(x['pred_norm_test'].keys()) for x in tmp_results]
        vocab_test = set(d[0]).intersection(*d[1:])
        d = [list(x['pred_norm_valid'].keys()) for x in tmp_results]
        vocab_valid = set(d[0]).intersection(*d[1:])

        for tmp_result in tmp_results:
            pred_norm_test = tmp_result.pop('pred_norm_test')
            pred_norm_valid = tmp_result.pop('pred_norm_valid')
            pred_org_test = tmp_result.pop('pred_org_test')
            pred_org_valid = tmp_result.pop('pred_org_valid')
            pred_norm_test = [int(pred_norm_test[k]) for k in vocab_test]

            pred_norm_valid = [int(pred_norm_valid[k]) for k in vocab_valid]
            pred_org_test = [int(pred_org_test[k]) for k in vocab_test]
            pred_org_valid = [int(pred_org_valid[k]) for k in vocab_valid]
            tmp_result['accuracy_test'] = sum(pred_norm_test) / len(pred_norm_test)
            tmp_result['accuracy_valid'] = sum(pred_norm_valid) / len(pred_norm_valid)
            tmp_result['accuracy_test_original'] = sum(pred_org_test) / len(pred_org_test)
            tmp_result['accuracy_valid_original'] = sum(pred_org_valid) / len(pred_org_valid)

            tmp_result['accuracy'] = (tmp_result['accuracy_test'] * len(test) +
                                      tmp_result['accuracy_valid'] * len(val)) / (len(val) + len(test))
            tmp_result['accuracy_original'] = (tmp_result['accuracy_test_original'] * len(test) +
                                               tmp_result['accuracy_valid_original'] * len(val)) / (len(val) + len(test))
            logging.info('\t * accuracy org : {}'.format(tmp_result['accuracy_original']))
            logging.info('\t * accuracy norm: {}'.format(tmp_result['accuracy']))

    df = pd.DataFrame(result)
    if opt.output_file is None:
        output_file = './eval/analogy/accuracy.glow_word_embedding.sat.csv'
    else:
        output_file = opt.output_file

    if os.path.exists(output_file):
        tmp = pd.read_csv(output_file, index_col=0)
        df = pd.concat([tmp, df])
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file)
    logging.info('result file exported to {}'.format(output_file))


# def main_bert():
#     main('glow_bert')


def main_word():
    main()

