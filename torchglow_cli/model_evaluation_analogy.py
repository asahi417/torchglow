""" Solve analogy task by word embedding model """
import logging
import argparse
import os
from glob import glob
from itertools import chain
from copy import deepcopy
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
    argument_parser.add_argument('--checkpoint-path', help='model checkpoint', default='./ckpt/{}/*'.format(model_type),
                                 type=str)
    argument_parser.add_argument('-o', '--output-dir', help='directory to export model weight file',
                                 default='./eval_output/{}/analogy_result.csv'.format(model_type), type=str)
    opt = argument_parser.parse_args()
    checkpoint_paths = glob(opt.checkpoint_path)
    result = []
    if model_type == 'glow_word_embedding':
        model_instance = torchglow.GlowWordEmbedding
    elif model_type == 'glow_bert':
        model_instance = torchglow.GlowBERT
    else:
        raise ValueError('unknown model type: {}'.format(model_type))
    base_prediction = torchglow.util.get_analogy_baseline()
    for n, checkpoint_path in enumerate(checkpoint_paths):
        logging.info('checkpoint: {}/{}'.format(n, len(checkpoint_paths)))
        # all intermediate epoch
        epochs = sorted([int(i.split('model.')[-1].replace('.pt', ''))
                         for i in glob('{}/model.*.pt'.format(checkpoint_path))])
        for e in epochs:
            logging.info('\t epoch: {}/{}'.format(e, epochs))
            model = model_instance(checkpoint_path=checkpoint_path, checkpoint_epoch=e)

            def get_word_pairs(word_pairs):
                all_pairs = list(chain(*[[o['stem']] + o['choice'] for o in word_pairs]))
                if model_type == 'glow_word_embedding':
                    return list(set(list(chain(*all_pairs))))
                    # elif model.data_format == 'relative':
                    #     all_pairs = [torchglow.util.word_pair_format(d) for d in all_pairs]
                    #     if model.vocab is not None:
                    #         return list(filter(lambda x: x in model.vocab, all_pairs))
                    #     return all_pairs
                    # else:
                    #     raise ValueError('unknown data format: {}'.format(model.data_format))
                elif model_type == 'glow_bert':
                    return all_pairs
                else:
                    raise ValueError('unknown model type: {}'.format(model_type))

            for i in DATA:
                tmp_result = deepcopy(model.parameter)
                tmp_result['epoch'] = e
                tmp_result['data'] = i
                val, test = torchglow.util.get_analogy_dataset(i)
                # cache embedding
                data = get_word_pairs(val + test)
                logging.info('\t cache embedding: {}'.format(i))
                vector, vector_original = model.embed(data, batch=opt.batch, return_original_embedding=True)
                latent_dict_normalized = {str(k): v for k, v in zip(data, vector)}
                latent_dict_original = {str(k): v for k, v in zip(data, vector_original)}
                logging.info('\t compute accuracy')

                def get_prediction(single_data, latent_dict):
                    """ OOV should only happen in `word` of `pair` format. """
                    if model_type == 'glow_word_embedding':
                        # if model.data_format == 'word':
                        diff_s = diff(latent_dict[single_data['stem'][0]], latent_dict[single_data['stem'][1]])
                        diff_c = [diff(latent_dict[a], latent_dict[b]) for a, b in single_data['choice']]
                        sim = [cos_similarity(diff_s, c) for c in diff_c]
                        # elif model.data_format == 'relative':
                        #     stem = torchglow.util.word_pair_format(single_data['stem'])
                        #     choice = [torchglow.util.word_pair_format(d) for d in single_data['choice']]
                        #     if stem not in model.vocab:
                        #         return None
                        #     sim = [cos_similarity(latent_dict[stem], latent_dict[c]) if c in latent_dict else -100
                        #            for c in choice]
                        # else:
                        #     raise ValueError('unknown data format: {}'.format(model.data_format))
                    elif model_type == 'glow_bert':
                        assert model.data_format == 'glow_bert', model.data_format
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
                    prediction_norm = [get_prediction(o, latent_dict_normalized) for o in data]
                    prediction_org = [get_prediction(o, latent_dict_original) for o in data]

                    tmp_result['oov_{}'.format(prefix)] = len([p for p in prediction_norm if p is None])
                    prediction_norm = [p if p is not None else base_prediction[i][prefix][n]
                                       for n, p in enumerate(prediction_norm)]
                    prediction_org = [p if p is not None else base_prediction[i][prefix][n]
                                      for n, p in enumerate(prediction_org)]
                    accuracy_norm = [o['answer'] == p for o, p in zip(data, prediction_norm)]
                    accuracy_org = [o['answer'] == p for o, p in zip(data, prediction_org)]
                    tmp_result['accuracy_{}'.format(prefix)] = sum(accuracy_norm)/len(accuracy_norm)
                    tmp_result['accuracy_{}_original'.format(prefix)] = sum(accuracy_org) / len(accuracy_org)
                tmp_result['accuracy'] = (tmp_result['accuracy_test'] * len(test) +
                                          tmp_result['accuracy_valid'] * len(val)) / (len(val) + len(test))
                tmp_result['accuracy_original'] = (tmp_result['accuracy_test_original'] * len(test) +
                                                   tmp_result['accuracy_valid_original'] * len(val)) / (len(val) + len(test))
                logging.info('\t accuracy org: {}'.format(tmp_result['accuracy_original']))
                logging.info('\t accuracy norm: {}'.format(tmp_result['accuracy']))
                result.append(tmp_result)

            del model

    df = pd.DataFrame(result)
    if os.path.exists(opt.output_dir):
        tmp = pd.read_csv(opt.output_dir, index_col=0)
        df = pd.concat([tmp, df])
    os.makedirs(os.path.dirname(opt.output_dir), exist_ok=True)
    df.to_csv(opt.output_dir)


def main_bert():
    main('glow_bert')


def main_word():
    main('glow_word_embedding')

