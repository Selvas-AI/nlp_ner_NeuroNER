# -*- coding: utf-8-*-
import argparse
import os
import pickle
import time
import warnings

import utils
from params import Configuration
from neuroner import NeuroNER
from src.metadata import Metadata

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameter', type=str, default='./parameters.ini')
    parser.add_argument('--mode', type=str, default='')
    cmd_arg = parser.parse_args()

    start_time = time.time()
    print('Init... ', end='', flush=True)
    parameters = Configuration(cmd_arg.parameter)

    if cmd_arg.mode != '':
        parameters['mode'] = cmd_arg.mode
    dataset_filepaths = utils.get_valid_dataset_filepaths(parameters['dataset_text_folder'])
    if parameters['mode'] == 'train':
        dataset_filepaths.pop('test', None)
        if 'train' not in dataset_filepaths or 'valid' not in dataset_filepaths:
            raise Exception('train data path empty')
    elif parameters['mode'] == 'test':
        dataset_filepaths.pop('train', None)
        dataset_filepaths.pop('valid', None)
        if 'test' not in dataset_filepaths:
            raise Exception('test data path empty')

    if parameters['mode'] == 'train':
        metadata = Metadata(parameters['dataset_text_folder'], dataset_filepaths, parameters['limit_word_size'],
                            parameters['remap_to_unk_count_threshold'])
    else:
        metadata = Metadata(parameters['pretrained_model_folder'], None, None, None)

    expanded_embedding = None
    if parameters['use_vocab_expansion'] and (parameters['mode'] == 'predict' or parameters['mode'] == 'test'):
        expanded_embedding_filepath = parameters['pretrained_model_folder'] + "/expanded_embedding.pickles"
        if not os.path.exists(expanded_embedding_filepath):
            raise Exception("expand embedding file not exist")
        with open(expanded_embedding_filepath, "rb") as f:
            expanded_embedding = pickle.load(f)

    neuroner = NeuroNER(parameters, metadata, expanded_embedding=expanded_embedding)
    print('done ({0:.2f} seconds)'.format(time.time() - start_time))

    if parameters['mode'] == 'train':
        neuroner.fit(dataset_filepaths)
    elif parameters['mode'] == 'test':
        neuroner.test(dataset_filepaths)
    elif parameters['mode'] == 'predict':
        print(neuroner.predict("내일 서울에서 5시반에 경찰청 원빈 과장이랑 같이 만납시다"))
    elif parameters['mode'] == 'vocab_expansion':
        neuroner.vocab_expansion()
    else:
        raise Exception("hello")
    neuroner.close()
    print('complete ({0:.2f} seconds)'.format(time.time() - start_time))


if __name__ == "__main__":
    main()
