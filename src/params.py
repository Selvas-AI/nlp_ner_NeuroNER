# -*- coding: utf-8-*-
import configparser
import os

import utils

CONLL_DEFAULT_LENGTH = 5
UNK = 'UNK'
PADDING_TOKEN_INDEX = 0
UNK_TOKEN_INDEX = PADDING_TOKEN_INDEX
PADDING_POS_INDEX = 0
PADDING_CHARACTER_INDEX = 0
PADDING_LABEL_INDEX = 0
LIMIT_SEQUENCE_LENGTH = 500
BREAK_STEP = 100000
DEFAULT_PARAMETER = {'pretrained_model_folder': '../trained_models/conll_2003_en',
                     'dataset_text_folder': '../data/conll2003/en',
                     'character_embedding_dimension': 25,
                     'character_lstm_hidden_state_dimension': 25,
                     'check_for_digits_replaced_with_zeros': True,
                     'check_for_lowercase': True,
                     'dropout_rate': 0.5,
                     'experiment_name': 'experiment',
                     'freeze_token_embeddings': False,
                     'gradient_clipping_value': 5.0,
                     'learning_rate': 0.005,
                     'load_only_pretrained_token_embeddings': False,
                     'main_evaluation_mode': 'conll',
                     'maximum_number_of_steps': 10000000000,
                     'number_of_cpu_threads': 8,
                     'number_of_gpus': 0,
                     'optimizer': 'sgd',
                     'output_folder': '../output',
                     'patience': 10,
                     'plot_format': 'pdf',
                     'remap_unknown_tokens_to_unk': True,
                     'token_embedding_dimension': 100,
                     'token_lstm_hidden_state_dimension': 100,
                     'token_pretrained_embedding_filepath': '../data/word_vectors/glove.6B.100d.txt',
                     'mode': 'train',
                     'use_vocab_expansion': False,
                     'use_character_lstm': True,
                     'use_crf': True,
                     'lstm_cell_type': 'lstm',
                     'batch_size': 1,
                     'limit_word_size': 0,
                     'remap_to_unk_count_threshold': 1,
                     'enable_tensorbord': False,
                     'tokenizer': 'pos'
                     }


class Configuration(dict):
    @staticmethod
    def __load_parameter(parameters_filepath):
        parameters = dict(DEFAULT_PARAMETER)

        if len(parameters_filepath) > 0:
            conf_parameters = configparser.ConfigParser()
            if not os.path.exists(parameters_filepath):
                raise Exception('parameters_filepath not exist : {}'.format(parameters_filepath))
            conf_parameters.read(parameters_filepath)
            nested_parameters = utils.convert_configparser_to_dictionary(conf_parameters)
            for k, v in nested_parameters.items():
                parameters.update(v)
        for k, v in parameters.items():
            v = str(v)
            if ',' in v:
                v = v.split(',')
            if k in ['character_embedding_dimension', 'character_lstm_hidden_state_dimension',
                     'token_embedding_dimension',
                     'token_lstm_hidden_state_dimension', 'patience', 'maximum_number_of_steps',
                     'maximum_training_time', 'number_of_cpu_threads', 'number_of_gpus', 'batch_size',
                     'limit_word_size',
                     'remap_to_unk_count_threshold']:
                if type(v) is list:
                    parameters[k] = [int(e) for e in v]
                else:
                    parameters[k] = int(v)
            elif k in ['dropout_rate', 'learning_rate', 'gradient_clipping_value']:
                parameters[k] = float(v)
            elif k in ['remap_unknown_tokens_to_unk', 'use_character_lstm', 'use_crf',
                       'check_for_lowercase', 'check_for_digits_replaced_with_zeros', 'use_vocab_expansion',
                       'freeze_token_embeddings', 'load_only_pretrained_token_embeddings',
                       'load_all_pretrained_token_embeddings', 'enable_tensorbord']:
                parameters[k] = utils.str2bool(v)
        parameters['ini_path'] = parameters_filepath
        return parameters

    def __init__(self, parameters_filepath):
        self.update(self.__load_parameter(parameters_filepath))
