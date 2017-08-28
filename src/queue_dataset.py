from __future__ import absolute_import
from __future__ import absolute_import
from __future__ import division
from __future__ import division
from __future__ import print_function
from __future__ import print_function

import collections
import glob
import os
import pickle
import re
import time

import utils
import utils_nlp
from batcher import Batcher
from brat2conll import brat_to_conll


class QueueDataset(object):
    UNK_TOKEN_INDEX = 0
    PADDING_TOKEN_INDEX = 0
    PADDING_POS_INDEX = 0
    PADDING_CHARACTER_INDEX = 0
    PADDING_LABEL_INDEX = 0
    UNK = 'UNK'

    def __init__(self, name='', verbose=False, debug=False):
        self.name = name
        self.verbose = verbose
        self.debug = debug

    def parse_data(self, data_path):
        token_count = collections.defaultdict(lambda: 0)
        label_count = collections.defaultdict(lambda: 0)
        character_count = collections.defaultdict(lambda: 0)
        number_of_pos_classes = 0
        file_list = list(glob.glob(data_path + "/*.conll"))
        sample_size = 0
        for file_path in file_list:
            with open(file_path, 'r', encoding='UTF-8') as f:
                file_content = f.read()
                sentences = file_content.split("\n\n")
                for sentence in sentences:
                    sentence = sentence.strip(" \n")
                    if len(sentence) == 0:
                        continue
                    lines = sentence.split("\n")
                    sample_size += 1
                    for line_raw in lines:
                        if '-DOCSTART-' in line_raw:
                            continue
                        line = line_raw.strip().split(' ')
                        token = str(line[0])
                        token = re.sub('\d+', '0', token)
                        pos = int(line[-3])
                        if pos < 0:
                            raise Exception
                        if pos > number_of_pos_classes:
                            number_of_pos_classes = pos
                        label = str(line[-1])
                        token_count[token] += 1
                        label_count[label] += 1
                        for character in token:
                            character_count[character] += 1
        return token_count, label_count, character_count, number_of_pos_classes + 1, sample_size

    def load_metadata(self, parameters, dataset_filepaths, token_to_vector):
        label_count = {}
        token_count = {}
        character_count = {}
        number_of_pos_classes = {}
        sample_size = {}
        for dataset_type, base in dataset_filepaths.items():
            token_count[dataset_type] = collections.defaultdict(lambda: 0)
            label_count[dataset_type] = collections.defaultdict(lambda: 0)
            character_count[dataset_type] = collections.defaultdict(lambda: 0)
            number_of_pos_classes[dataset_type] = 0
            if not os.path.exists(base):
                continue
            pk_path = os.path.join(os.path.abspath(base), dataset_type + ".pickles")
            brat_to_conll(base)
            if os.path.exists(pk_path):
                with open(pk_path, "rb") as f:
                    loaded_pk = pickle.load(f)
                    label_count[dataset_type].update(loaded_pk['label_count'])
                    token_count[dataset_type].update(loaded_pk['token_count'])
                    character_count[dataset_type].update(loaded_pk['character_count'])
                    number_of_pos_classes[dataset_type] = loaded_pk['number_of_pos_classes']
                    sample_size[dataset_type] = loaded_pk['sample_size']
            else:
                token_count[dataset_type], label_count[dataset_type], character_count[
                    dataset_type], number_of_pos_classes[dataset_type], sample_size[dataset_type] = self.parse_data(base)
                pickle.dump({'token_count': dict(token_count[dataset_type]), 'label_count': dict(label_count[dataset_type]),
                             'character_count': dict(character_count[dataset_type]),
                             'number_of_pos_classes': number_of_pos_classes[dataset_type],
                             'sample_size' : sample_size[dataset_type]
                             }, open(pk_path, "wb"))

        self.number_of_pos_classes = max(list(number_of_pos_classes.values()))
        self.tokens_mapped_to_unk = []
        self.sample_size = sample_size

        all_tokens_in_pretraining_dataset = []
        all_characters_in_pretraining_dataset = []
        if parameters['use_pretrained_model']:
            pretraining_dataset = pickle.load(
                open(os.path.join(parameters['pretrained_model_folder'], 'dataset.pickle'), 'rb'))
            all_tokens_in_pretraining_dataset = pretraining_dataset.index_to_token.values()
            all_characters_in_pretraining_dataset = pretraining_dataset.index_to_character.values()

        remap_to_unk_count_threshold = 1

        token_count['all'] = {}
        for token in list(token_count['train'].keys()) + list(token_count['valid'].keys()) + list(
                token_count['test'].keys()) + list(token_count['deploy'].keys()):
            token_count['all'][token] = token_count['train'][token] + token_count['valid'][token] + token_count['test'][
                token] + token_count['deploy'][token]

        if parameters['load_all_pretrained_token_embeddings']:
            for token in token_to_vector:
                if token not in token_count['all']:
                    token_count['all'][token] = -1
                    token_count['train'][token] = -1
            for token in all_tokens_in_pretraining_dataset:
                if token not in token_count['all']:
                    token_count['all'][token] = -1
                    token_count['train'][token] = -1

        character_count['all'] = {}
        for character in list(character_count['train'].keys()) + list(character_count['valid'].keys()) + list(
                character_count['test'].keys()) + list(character_count['deploy'].keys()):
            character_count['all'][character] = character_count['train'][character] + character_count['valid'][
                character] + character_count['test'][character] + character_count['deploy'][character]

        for character in all_characters_in_pretraining_dataset:
            if character not in character_count['all']:
                character_count['all'][character] = -1
                character_count['train'][character] = -1

        label_count['all'] = {}
        for character in list(label_count['train'].keys()) + list(label_count['valid'].keys()) + list(
                label_count['test'].keys()) + list(label_count['deploy'].keys()):
            label_count['all'][character] = label_count['train'][character] + label_count['valid'][character] + \
                                            label_count['test'][character] + label_count['deploy'][character]

        token_count['all'] = utils.order_dictionary(token_count['all'], 'value_key', reverse=True)
        label_count['all'] = utils.order_dictionary(label_count['all'], 'key', reverse=False)
        character_count['all'] = utils.order_dictionary(character_count['all'], 'value', reverse=True)

        print('all token_count size is %d' % len(token_count['all']))
        if parameters['limit_word_size'] > 0 and len(token_count['all']) > parameters['limit_word_size']:
            limited_dict = dict(list(token_count['all'].items())[:parameters['limit_word_size']])
            print('token_count is limited to %d' % parameters['limit_word_size'])

        token_to_index = {}
        token_to_index[self.UNK] = self.UNK_TOKEN_INDEX
        iteration_number = 0
        number_of_unknown_tokens = 0
        for token, count in token_count['all'].items():
            if iteration_number == self.UNK_TOKEN_INDEX: iteration_number += 1

            if ('limited_dict' in locals() and token not in limited_dict) or \
                parameters['remap_unknown_tokens_to_unk'] == 1 and \
                    (token_count['train'][token] == 0 or \
                             parameters['load_only_pretrained_token_embeddings']) and \
                    not utils_nlp.is_token_in_pretrained_embeddings(token, token_to_vector, parameters) and \
                            token not in all_tokens_in_pretraining_dataset:
                token_to_index[token] = self.UNK_TOKEN_INDEX
                number_of_unknown_tokens += 1
                self.tokens_mapped_to_unk.append(token)
            else:
                token_to_index[token] = iteration_number
                iteration_number += 1

        infrequent_token_indices = []
        for token, count in token_count['train'].items():
            if 0 < count <= remap_to_unk_count_threshold:
                infrequent_token_indices.append(token_to_index[token])

        # Ensure that both B- and I- versions exist for each label
        labels_without_bio = set()
        for label in label_count['all'].keys():
            new_label = utils_nlp.remove_bio_from_label_name(label)
            labels_without_bio.add(new_label)
        for label in labels_without_bio:
            if label == 'O':
                continue
            if parameters['tagging_format'] == 'bioes':
                prefixes = ['B-', 'I-', 'E-', 'S-']
            else:
                prefixes = ['B-', 'I-']
            for prefix in prefixes:
                l = prefix + label
                if l not in label_count['all']:
                    label_count['all'][l] = 0
        label_count['all'] = utils.order_dictionary(label_count['all'], 'key', reverse=False)

        if parameters['use_pretrained_model']:
            # Make sure labels are compatible with the pretraining dataset.
            for label in label_count['all']:
                if label not in pretraining_dataset.label_to_index:
                    raise AssertionError("The label {0} does not exist in the pretraining dataset. ".format(label) +
                                         "Please ensure that only the following labels exist in the dataset: {0}".format(
                                             ', '.join(self.unique_labels)))
            label_to_index = pretraining_dataset.label_to_index.copy()
        else:
            label_to_index = {}
            iteration_number = 0
            for label, count in label_count['all'].items():
                label_to_index[label] = iteration_number
                iteration_number += 1

        self.unique_labels = sorted(list(label_to_index.keys()))

        character_to_index = {}
        iteration_number = 0
        for character, count in character_count['all'].items():
            if iteration_number == self.PADDING_CHARACTER_INDEX: iteration_number += 1
            character_to_index[character] = iteration_number
            iteration_number += 1

        token_to_index = utils.order_dictionary(token_to_index, 'value', reverse=False)
        index_to_token = utils.reverse_dictionary(token_to_index)
        if parameters['remap_unknown_tokens_to_unk'] == 1: index_to_token[self.UNK_TOKEN_INDEX] = self.UNK

        label_to_index = utils.order_dictionary(label_to_index, 'value', reverse=False)
        index_to_label = utils.reverse_dictionary(label_to_index)
        character_to_index = utils.order_dictionary(character_to_index, 'value', reverse=False)
        index_to_character = utils.reverse_dictionary(character_to_index)

        self.token_to_index = token_to_index
        self.index_to_token = index_to_token
        self.index_to_character = index_to_character
        self.character_to_index = character_to_index
        self.index_to_label = index_to_label
        self.label_to_index = label_to_index
        self.number_of_classes = max(self.index_to_label.keys()) + 1
        self.vocabulary_size = max(self.index_to_token.keys()) + 1
        self.alphabet_size = max(self.index_to_character.keys()) + 1
        self.unique_labels_of_interest = list(self.unique_labels)
        self.unique_labels_of_interest.remove('O')
        self.unique_label_indices_of_interest = []
        for lab in self.unique_labels_of_interest:
            self.unique_label_indices_of_interest.append(label_to_index[lab])
        self.infrequent_token_indices = infrequent_token_indices

        self.max_token_size = 500

    def load_dataset(self, dataset_filepaths, parameters, token_to_vector=None):
        '''
        dataset_filepaths : dictionary with keys 'train', 'valid', 'test', 'deploy'
        '''
        start_time = time.time()
        print('Load dataset... ', end='', flush=True)
        if parameters['token_pretrained_embedding_filepath'] != '':
            if token_to_vector == None:
                token_to_vector = utils_nlp.load_pretrained_token_embeddings(parameters)
        else:
            token_to_vector = {}

        self.load_metadata(parameters, dataset_filepaths, token_to_vector)

        data_queue = {}
        for dataset_type, base in dataset_filepaths.items():
            if os.path.exists(base):
                data_queue[dataset_type] = Batcher(self, base, parameters['batch_size'],
                                        is_train=True if dataset_type == 'train' else False)

        self.data_queue = data_queue

        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))

        return token_to_vector


if __name__ == "__main__":
    # First, we are going to generate a single file which contains both training images and labels
    #  in standard tensorflow file format (TFRecords), this is simple
    base = r'D:\tech\entity\NeuroNER\data\exobrain'
    parameters = {}
    parameters['token_pretrained_embedding_filepath'] = ''
    parameters['batch_size'] = 10
    ds = QueueDataset()
    ds.load_dataset(base, parameters)
