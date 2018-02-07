# -*- coding: utf-8-*-
import collections
import glob
import os
import pickle

import preprocess
import utils
from params import CONLL_DEFAULT_LENGTH, UNK_TOKEN_INDEX, PADDING_LABEL_INDEX, PADDING_CHARACTER_INDEX


class Metadata(dict):
    @staticmethod
    def load_metadata(data_path, limit_word_size, remap_to_unk_count_threshold):
        token_count = collections.defaultdict(lambda: 0)
        label_count = collections.defaultdict(lambda: 0)
        character_count = collections.defaultdict(lambda: 0)
        num_of_extendeds = None
        token_to_index = collections.defaultdict(lambda: UNK_TOKEN_INDEX)
        label_to_index = collections.defaultdict(lambda: PADDING_LABEL_INDEX)
        character_to_index = collections.defaultdict(lambda: PADDING_CHARACTER_INDEX)

        file_list = list(glob.glob(data_path + "/*.txt"))

        for file_path in file_list:
            with open(file_path, 'r', encoding='UTF-8') as f:
                for line in f:
                    line = line.strip(" \n")
                    if len(line) == 0 or '-DOCSTART-' in line:
                        continue
                    splitted = line.split(' ')
                    token = str(splitted[0])
                    token = preprocess.normalize_token(token)
                    label = str(splitted[-1])

                    extended = []
                    if len(splitted) > CONLL_DEFAULT_LENGTH:
                        diff_len = len(splitted) - CONLL_DEFAULT_LENGTH
                        for idx in range(diff_len):
                            extended.append(int(splitted[CONLL_DEFAULT_LENGTH - 1 + idx]))

                    if num_of_extendeds == None:
                        num_of_extendeds = [0] * len(extended)

                    for idx, ex in enumerate(extended):
                        num_of_extendeds[idx] = max(num_of_extendeds[idx], ex)

                    token_count[token] += 1
                    label_count[label] += 1
                    for character in token:
                        character_count[character] += 1

        print('all token size is %d' % len(token_count))
        if limit_word_size > 0 and len(token_count) > limit_word_size:
            limited_dict = dict(list(token_count['all'].items())[:limit_word_size])
            print('token_count is limited to %d' % limit_word_size)

        # token_to_index
        iteration_number = 0
        for token, count in token_count.items():
            if iteration_number == UNK_TOKEN_INDEX: iteration_number += 1
            if 'limited_dict' not in locals() or token in limited_dict:
                token_to_index[token] = iteration_number
                iteration_number += 1
        num_of_token = iteration_number

        # label_to_index
        iteration_number = 0
        for label, count in label_count.items():
            label_to_index[label] = iteration_number
            iteration_number += 1
        num_of_label = iteration_number

        # character_to_index
        iteration_number = 0
        for label, count in character_count.items():
            character_to_index[label] = iteration_number
            iteration_number += 1
        num_of_char = iteration_number

        num_of_extendeds = [ex + 1 for ex in num_of_extendeds]

        # infrequent_token_indices
        infrequent_token_indices = []
        for token, count in token_count.items():
            if 0 < count <= remap_to_unk_count_threshold:
                infrequent_token_indices.append(token_to_index[token])
        return dict(token_to_index), dict(label_to_index), dict(
            character_to_index), num_of_token, num_of_label, num_of_char, num_of_extendeds, infrequent_token_indices

    def __init__(self, dataset_root, dataset_filepaths, limit_word_size, remap_to_unk_count_threshold):
        meta_path = os.path.join(dataset_root + "/meta.pickles")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                rawdata = pickle.load(f)

        else:
            token_to_index, label_to_index, character_to_index, num_of_token, num_of_label, num_of_char, num_of_extendeds, infrequent_token_indices = self.load_metadata(
                dataset_filepaths['train'], limit_word_size, remap_to_unk_count_threshold)
            rawdata = {
                'token_to_index': token_to_index,
                'label_to_index': label_to_index,
                'character_to_index': character_to_index,
                'index_to_token': utils.reverse_dictionary(token_to_index),
                'index_to_label': utils.reverse_dictionary(label_to_index),
                'index_to_character': utils.reverse_dictionary(character_to_index),
                'num_of_token': num_of_token,
                'num_of_label': num_of_label,
                'num_of_char': num_of_char,
                'num_of_extendeds': num_of_extendeds,
                'infrequent_token_indices': infrequent_token_indices
            }
        self.update(rawdata)
        self['prev_num_of_token'] = self['num_of_token']
        self['prev_num_of_char'] = self['num_of_char']
        del rawdata

    def patch(self, dataset_filepaths, limit_word_size, remap_to_unk_count_threshold):
        token_to_index, label_to_index, character_to_index, num_of_token, num_of_label, num_of_char, num_of_extendeds, infrequent_token_indices = self.load_metadata(
            dataset_filepaths['train'], limit_word_size, remap_to_unk_count_threshold)

        iteration_number = self['num_of_token']
        for token, value in token_to_index.items():
            if token in self['token_to_index']:
                continue
            self['token_to_index'][token] = iteration_number
            iteration_number += 1
        self['num_of_token'] = iteration_number

        iteration_number = self['num_of_label']
        for label, value in label_to_index.items():
            if label in self['label_to_index']:
                continue
            self['label_to_index'][label] = iteration_number
            iteration_number += 1
        self['num_of_label'] = iteration_number
        iteration_number = self['num_of_char']
        for char, value in character_to_index.items():
            if char in self['character_to_index']:
                continue
            self['character_to_index'][char] = iteration_number
            iteration_number += 1
        self['num_of_char'] = iteration_number

        self['index_to_token'] = utils.reverse_dictionary(self['token_to_index'])
        self['index_to_label'] = utils.reverse_dictionary(self['label_to_index'])
        self['index_to_character'] = utils.reverse_dictionary(self['character_to_index'])

        index_to_token = utils.reverse_dictionary(token_to_index)
        for index in infrequent_token_indices:
            token = index_to_token[index]
            if token not in self['infrequent_token_indices']:
                self['infrequent_token_indices'].append(self['token_to_index'][token])
        for idx, value in enumerate(num_of_extendeds):
            self['num_of_extendeds'][idx] = max(self['num_of_extendeds'][idx], value)

    def write(self, target_path):
        rawdata = {
            'token_to_index': self['token_to_index'],
            'label_to_index': self['label_to_index'],
            'character_to_index': self['character_to_index'],
            'index_to_token': self['index_to_token'],
            'index_to_label': self['index_to_label'],
            'index_to_character': self['index_to_character'],
            'num_of_token': self['num_of_token'],
            'num_of_label': self['num_of_label'],
            'num_of_char': self['num_of_char'],
            'num_of_extendeds': self['num_of_extendeds'],
            'infrequent_token_indices': self['infrequent_token_indices'],
        }

        meta_path = os.path.join(target_path + "/meta.pickles")
        pickle.dump(rawdata, open(meta_path, "wb"))
