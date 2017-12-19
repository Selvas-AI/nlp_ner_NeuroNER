import numpy as np

import utils


class BatchInputGenerator(object):
    def __init__(self, dataset, dataset_type, batch_size):
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.batch_size = int(batch_size)
        self.data_count = 0
        self.permutation_index = None

    def begin(self, random=True):
        self.data_count = 0
        if random:
            self.permutation_index = np.random.permutation(len(self.dataset.token_indices[self.dataset_type]))
        else:
            self.permutation_index = range(len(self.dataset.token_indices[self.dataset_type]))

    def size(self):
        return len(self.dataset.token_indices[self.dataset_type])

    def next(self):
        if self.data_count >= len(self.permutation_index):
            return None
        batch_index = []
        for i in range(self.batch_size):
            index = i + self.data_count
            if index >= len(self.permutation_index):
                index %= len(self.permutation_index)
            batch_index.append(self.permutation_index[index])
        self.data_count += self.batch_size
        return self._create_input_batch(batch_index)

    def _create_input_batch(self, batch_index):
        batch_data = {}
        max_token_size = 0
        for index in batch_index:
            max_token_size = max(max_token_size, len(self.dataset.token_indices[self.dataset_type][index]))

        batch_data['token_indices'] = []
        if self.dataset_type == 'train':
            for index in batch_index:
                token_indices = self.dataset.token_indices[self.dataset_type][index]
                for i, token_index in enumerate(self.dataset.token_indices[self.dataset_type][index]):
                    if token_index in self.dataset.infrequent_token_indices and np.random.uniform() < 0.5:
                        token_indices[i] = self.dataset.token_to_index[self.dataset.UNK]
                batch_data['token_indices'].append(
                    utils.pad_list(token_indices, max_token_size,
                                   self.dataset.PADDING_TOKEN_INDEX))
        else:
            for index in batch_index:
                batch_data['token_indices'].append(
                    utils.pad_list(self.dataset.token_indices[self.dataset_type][index], max_token_size,
                                   self.dataset.PADDING_TOKEN_INDEX))

        batch_data['space_indices'] = []
        for index in batch_index:
            batch_data['space_indices'].append(
                utils.pad_list(self.dataset.space_indices[self.dataset_type][index], max_token_size, 0))

        batch_data['token_morpheme_indices'] = []
        for index in batch_index:
            batch_data['token_morpheme_indices'].append(
                utils.pad_list(self.dataset.token_morpheme_indices[self.dataset_type][index], max_token_size,
                               [self.dataset.PADDING_POS_INDEX] * self.dataset.number_of_pos_classes))

        batch_data['label_vector_indices'] = []
        for index in batch_index:
            batch_data['label_vector_indices'].append(
                utils.pad_list(self.dataset.label_vector_indices[self.dataset_type][index], max_token_size,
                               [self.dataset.PADDING_LABEL_INDEX] * self.dataset.number_of_classes))

        batch_data['label_indices'] = []
        for index in batch_index:
            batch_data['label_indices'].append(
                utils.pad_list(self.dataset.label_indices[self.dataset_type][index], max_token_size,
                               self.dataset.PADDING_LABEL_INDEX))

        max_char_size = 0
        for index in batch_index:
            max_char_size = max(max_char_size, len(self.dataset.character_indices_padded[self.dataset_type][index][0]))

        batch_data['character_indices_padded'] = []
        for index in batch_index:
            word_list = []
            for word in self.dataset.character_indices_padded[self.dataset_type][index]:
                word_list.append(utils.pad_list(word, max_char_size,
                                                self.dataset.PADDING_CHARACTER_INDEX))
            batch_data['character_indices_padded'].append(
                utils.pad_list(word_list, max_token_size,
                               [self.dataset.PADDING_CHARACTER_INDEX] * max_char_size))

        batch_data['token_lengths'] = []
        for index in batch_index:
            batch_data['token_lengths'].append(
                utils.pad_list(self.dataset.token_lengths[self.dataset_type][index], max_token_size,
                               0))

        batch_data['seq_lengths'] = []
        for index in batch_index:
            batch_data['seq_lengths'].append(len(self.dataset.token_indices[self.dataset_type][index]))

        batch_data['batch_index'] = batch_index
        return batch_data
