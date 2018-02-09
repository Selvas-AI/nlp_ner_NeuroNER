# -*- coding: utf-8-*-
import argparse
import os
import pickle
import time
import warnings

import utils
from params import Configuration
from neuroner import NeuroNER
from metadata import Metadata
from oktpy.twitter import TwitterMorphManager

warnings.filterwarnings('ignore')
TwitterMorphManager().morph_analyzer.pos("안녕하세요")

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
        #dataset_filepaths.pop('valid', None)
        if len(dataset_filepaths) == 0:
            raise Exception('test data path empty')

    if parameters['mode'] == 'train':
        if parameters['load_pretrained_model']:
            metadata = Metadata(parameters['pretrained_model_folder'], None, None, None)
            metadata.patch(dataset_filepaths, parameters['limit_word_size'], parameters['remap_to_unk_count_threshold'])
        else:
            metadata = Metadata(parameters['dataset_text_folder'], dataset_filepaths, parameters['limit_word_size'],
                                parameters['remap_to_unk_count_threshold'])
            metadata.write(parameters['dataset_text_folder'])
    else:
        metadata = Metadata(parameters['pretrained_model_folder'], None, None, None)

    neuroner = NeuroNER(parameters, metadata)
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
