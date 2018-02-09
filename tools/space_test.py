# -*- coding: utf-8-*-
import argparse
import time
import warnings

from oktpy.twitter import TwitterMorphManager

import utils
from neuroner import NeuroNER
from params import Configuration
from src.metadata import Metadata

warnings.filterwarnings('ignore')
TwitterMorphManager().morph_analyzer.pos("안녕하세요")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameter', type=str, default=r"D:\tech\model\parameters.ini")
    parser.add_argument('--mode', type=str, default='predict')
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

    neuroner = NeuroNER(parameters, metadata)
    print('done ({0:.2f} seconds)'.format(time.time() - start_time))

    if parameters['mode'] == 'train':
        neuroner.fit(dataset_filepaths)
    elif parameters['mode'] == 'test':
        neuroner.test(dataset_filepaths)
    elif parameters['mode'] == 'predict':
        percent_list = ["0", "10", "30", "50", "100"]
        # percent_list = ["30", "50", "100"]
        for percent in percent_list:
            tic = time.time()
            total_line = 0
            with open(r"D:\tech\space_data\error%s.output.txt" % percent, "w", encoding='utf-8') as outfile:
                with open(r"D:\tech\space_data\error%s.input.txt" % percent, "r", encoding='utf-8') as f:
                    for line in f:
                        line = line.strip("\n")
                        total_line += len(line)
                        try:
                            tokens, extended_sequence, tags, score = neuroner.predict(line)
                        except Exception as e:
                            print(str(e))
                            outfile.write(line + "\n")
                            continue
                        output = ""
                        for idx, (tok, tag, ex_seq) in enumerate(zip(tokens, tags, extended_sequence)):
                            output += tok
                            if len(tokens) - 1 != idx and (ex_seq[0] == 1 or tag != 'O'):
                                # if len(tokens) - 1 != idx and tag != 'O':
                                output += " "
                        outfile.write(output + "\n")
            toc = time.time()

            print('complete ({0:.5f} seconds)'.format((toc - tic) / (total_line / 50)))

    elif parameters['mode'] == 'vocab_expansion':
        neuroner.vocab_expansion()
    else:
        raise Exception("hello")
    neuroner.close()
    print('complete ({0:.2f} seconds)'.format(time.time() - start_time))


if __name__ == "__main__":
    main()
