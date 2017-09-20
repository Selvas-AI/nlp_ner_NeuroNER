# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Batch reader to seq2seq attention model, with bucketing support."""
import glob
import re
import time
from collections import namedtuple
from multiprocessing import Process, Queue
from random import shuffle

import numpy as np
import tensorflow as tf

import utils

ModelInput = namedtuple('ModelInput',
                        'token_indices character_indices label_indices pos_sequence space_sequence conll token_lengths sequence_length')

BUCKET_CACHE_BATCH = 20
QUEUE_NUM_BATCH = 1000


class Batcher(object):
    """Batch reader with shuffling and bucketing support."""

    def __init__(self, qd, data_path, batch_size, is_train, bucketing=False, truncate_input=False):
        """Batcher constructor.
        Args:
          data_path: tf.Example filepattern.
          vocab: Vocabulary.
          hps: Seq2SeqAttention model hyperparameters.
          article_key: article feature key in tf.Example.
          abstract_key: abstract feature key in tf.Example.
          max_article_sentences: Max number of sentences used from article.
          max_abstract_sentences: Max number of sentences used from abstract.
          bucketing: Whether bucket articles of similar length into the same batch.
          truncate_input: Whether to truncate input that is too long. Alternative is
            to discard such examples.
        """
        self._qd = qd
        self._data_path = data_path
        self._batch_size = batch_size
        self._is_train = is_train
        if self._is_train:
            input_thread_num = 2
            bucket_thread_num = 4
        else:
            input_thread_num = 1
            bucket_thread_num = 1
        self._bucketing = bucketing
        self._truncate_input = truncate_input
        self._input_queue = Queue(QUEUE_NUM_BATCH * self._batch_size)
        self._bucket_input_queue = Queue(QUEUE_NUM_BATCH)
        self._input_threads = None
        self._bucketing_threads = None
        self._watch_thread = None
        self._input_threads = []
        for _ in range(input_thread_num):
            self._input_threads.append(Process(target=self._FillInputQueue))
            self._input_threads[-1].daemon = True
            self._input_threads[-1].start()
        self._bucketing_threads = []
        for _ in range(bucket_thread_num):
            self._bucketing_threads.append(Process(target=self._FillBucketInputQueue))
            self._bucketing_threads[-1].daemon = True
            self._bucketing_threads[-1].start()

            # self._watch_thread = Process(target=self._WatchThreads)
            # self._watch_thread.daemon = True
            # self._watch_thread.start()

    def next(self):
        batch_data = self._bucket_input_queue.get()
        return (batch_data['seq_lengths'], batch_data['label_indices'], batch_data['token_indices'],
                batch_data['pos_sequence'], batch_data['space_sequence'], batch_data['token_lengths'],
                batch_data['character_indices'], batch_data['conll'])

    def _FillInputQueue(self):
        number_of_pos_classes = 0
        file_list = list(glob.glob(self._data_path + "/*.conll"))
        while True:
            if self._is_train:
                shuffle(file_list)
            for file_path in file_list:
                with open(file_path, 'r', encoding='UTF-8') as f:
                    file_content = f.read()
                    sentences = file_content.split("\n\n")
                    for sentence in sentences:
                        token_sequence = []
                        label_sequence = []
                        pos_sequence = []
                        space_sequence = []
                        original_conll = ''

                        sentence = sentence.strip(" \n")
                        if len(sentence) == 0:
                            continue
                        lines = sentence.split("\n")
                        for line_raw in lines:
                            if '-DOCSTART-' in line_raw:
                                continue
                            line = line_raw.strip().split(' ')
                            token = str(line[0])
                            token = re.sub('\d+', '0', token)
                            pos = int(line[-3])
                            sp = int(line[-2])
                            if pos < 0:
                                raise Exception
                            if pos > number_of_pos_classes:
                                number_of_pos_classes = pos
                            label = str(line[-1])

                            token_sequence.append(token)
                            label_sequence.append(label)
                            pos_sequence.append(pos)
                            space_sequence.append(sp)
                            original_conll += line_raw + "\n"
                        if len(token_sequence) > 0:
                            element = self._encode(label_sequence, pos_sequence, space_sequence, token_sequence,
                                                   original_conll)
                            if element is not None: self._input_queue.put(element)

    def _encode(self, label_sequence, pos_sequence, space_sequence, token_sequence, original_conll):
        label_indices = []
        for label in label_sequence:
            if label in self._qd.label_to_index:
                label_indices.append(self._qd.label_to_index[label])
            else:
                label_indices.append(self._qd.PADDING_LABEL_INDEX)

        token_indices = []
        for token in token_sequence:
            if token in self._qd.token_to_index:
                token_indices.append(self._qd.token_to_index[token])
            else:
                token_indices.append(self._qd.UNK_TOKEN_INDEX)

        character_indices = []
        for token in token_sequence:
            current_token_character_indices = []
            for character in token:
                if character in self._qd.character_to_index:
                    current_token_character_indices.append(self._qd.character_to_index[character])
                else:
                    current_token_character_indices.append(self._qd.PADDING_CHARACTER_INDEX)
            # current_token_character_indices = utils.pad_list(current_token_character_indices, self.MAX_CHARACTER_LEN, self.PADDING_CHARACTER_INDEX)
            character_indices.append(current_token_character_indices)
        token_lengths = [len(token) for token in token_sequence]
        sequence_length = len(token_indices)
        return ModelInput(token_indices, character_indices, label_indices, pos_sequence, space_sequence, original_conll,
                          token_lengths, sequence_length)

    def _FillBucketInputQueue(self):
        """Fill bucketed batches into the bucket_input_queue."""
        while True:
            inputs = []
            for _ in range(self._batch_size * BUCKET_CACHE_BATCH):
                inputs.append(self._input_queue.get())
            if self._is_train:
                inputs = sorted(inputs, key=lambda inp: inp.sequence_length)

            batches = []
            for i in range(0, len(inputs), self._batch_size):
                batch_data = {}
                current = inputs[i:i + self._batch_size]
                max_token_size = max([len(element.token_indices) for element in current])

                batch_data['token_indices'] = []
                for element in current:
                    if self._is_train:
                        for j, token_index in enumerate(element.token_indices):
                            if token_index in self._qd.infrequent_token_indices and np.random.uniform() < 0.5:
                                element.token_indices[j] = self._qd.token_to_index['UNK']
                    batch_data['token_indices'].append(
                        utils.pad_list(element.token_indices, max_token_size, self._qd.PADDING_TOKEN_INDEX))

                batch_data['space_sequence'] = []
                for element in current:
                    batch_data['space_sequence'].append(utils.pad_list(element.space_sequence, max_token_size, 0))

                batch_data['pos_sequence'] = []
                for element in current:
                    batch_data['pos_sequence'].append(utils.pad_list(element.pos_sequence, max_token_size, 0))

                batch_data['label_indices'] = []
                for element in current:
                    batch_data['label_indices'].append(utils.pad_list(element.label_indices, max_token_size, -1000))

                max_char_size = 0
                for element in current:
                    for word in element.character_indices:
                        max_char_size = max(max_char_size, len(word))

                batch_data['character_indices'] = []
                for element in current:
                    word_list = []
                    for word in element.character_indices:
                        word_list.append(utils.pad_list(word, max_char_size, self._qd.PADDING_CHARACTER_INDEX))
                    batch_data['character_indices'].append(
                        utils.pad_list(word_list, max_token_size, [self._qd.PADDING_CHARACTER_INDEX] * max_char_size))

                batch_data['token_lengths'] = []
                for element in current:
                    batch_data['token_lengths'].append(
                        utils.pad_list(element.token_lengths, max_token_size, 0))

                batch_data['seq_lengths'] = []
                for element in current:
                    batch_data['seq_lengths'].append(element.sequence_length)

                batch_data['conll'] = []
                for element in current:
                    batch_data['conll'].append(element.conll)

                batches.append(batch_data)

            if self._is_train:
                shuffle(batches)
            for b in batches:
                self._bucket_input_queue.put(b)

    def _WatchThreads(self):
        """Watch the daemon input threads and restart if dead."""
        while True:
            time.sleep(60)
            input_threads = []
            for t in self._input_threads:
                if t.is_alive():
                    input_threads.append(t)
                else:
                    tf.logging.error('Found input thread dead.')
                    new_t = Process(target=self._FillInputQueue)
                    input_threads.append(new_t)
                    input_threads[-1].daemon = True
                    input_threads[-1].start()
            self._input_threads = input_threads

            bucketing_threads = []
            for t in self._bucketing_threads:
                if t.is_alive():
                    bucketing_threads.append(t)
                else:
                    tf.logging.error('Found bucketing thread dead.')
                    new_t = Process(target=self._FillBucketInputQueue)
                    bucketing_threads.append(new_t)
                    bucketing_threads[-1].daemon = True
                    bucketing_threads[-1].start()
            self._bucketing_threads = bucketing_threads

    def close(self):
        self.__del__()

    def __del__(self):
        for t in self._input_threads:
            t.terminate()
            t.join()
        for t in self._bucketing_threads:
            t.terminate()
            t.join()
        self._input_queue.close()
        self._bucket_input_queue.close()
