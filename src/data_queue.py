# -*- coding: utf-8-*-
import glob
import time
from multiprocessing import Process
from multiprocessing import Queue as ProcessQueue
from queue import Queue as ThreadQueue
from random import shuffle
from threading import Thread

import jpype
import tensorflow as tf

import preprocess
from params import CONLL_DEFAULT_LENGTH


class DataQueue(object):
    """Batch reader with shuffling and bucketing support."""

    def __init__(self, metadata, data_path, batch_size, is_train,
                 bucketing=False, truncate_input=False, use_process=False,
                 expanded_embedding=None):
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
        self._use_process = use_process
        if self._use_process:
            Worker = Process
            Queue = ProcessQueue
        else:
            Worker = Thread
            Queue = ThreadQueue
        self._metadata = metadata
        self._data_path = data_path
        self._batch_size = batch_size
        self._is_train = is_train
        self._expanded_embedding = expanded_embedding
        if self._is_train:
            input_thread_num = 2
            bucket_thread_num = 1
            self._bucket_cache_batch = 20
            self._queue_num_batch = 500
        else:
            input_thread_num = 1
            bucket_thread_num = 1
            self._bucket_cache_batch = 1
            self._queue_num_batch = 30
        self._bucketing = bucketing
        self._truncate_input = truncate_input
        self._input_queue = Queue(self._queue_num_batch * self._batch_size)
        self._bucket_input_queue = Queue(self._queue_num_batch)
        self._input_threads = None
        self._bucketing_threads = None
        self._watch_thread = None
        self._input_threads = []

        for _ in range(input_thread_num):
            self._input_threads.append(Worker(target=self._FillInputQueue))
            self._input_threads[-1].daemon = True
            self._input_threads[-1].start()
        self._bucketing_threads = []
        for _ in range(bucket_thread_num):
            self._bucketing_threads.append(Worker(target=self._FillBucketInputQueue))
            self._bucketing_threads[-1].daemon = True
            self._bucketing_threads[-1].start()

    def next(self):
        return self._bucket_input_queue.get()

    def _FillInputQueue(self):
        if not self._use_process:
            jpype.attachThreadToJVM()
        file_list = list(glob.glob(self._data_path + "/*.txt"))
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
                        extended_sequence = []
                        conll_txt = ''

                        sentence = sentence.strip(" \n")
                        if len(sentence) == 0:
                            continue
                        lines = sentence.split("\n")
                        for line_raw in lines:
                            if '-DOCSTART-' in line_raw:
                                continue
                            line = line_raw.strip().split(' ')

                            token = str(line[0])
                            label = str(line[-1])
                            token_sequence.append(token)
                            label_sequence.append(label)
                            conll_txt += line_raw + "\n"

                            extended = []
                            if len(line) > CONLL_DEFAULT_LENGTH:
                                diff_len = len(line) - CONLL_DEFAULT_LENGTH
                                for idx in range(diff_len):
                                    extended.append(int(line[CONLL_DEFAULT_LENGTH - 1 + idx]))
                            else:
                                # todo 확장 feature가 없을 경우 예외처리가 까다로워서 없으면 0이라도 채워넣음
                                extended = [0]
                            extended_sequence.append(extended)

                        if len(token_sequence) > 0:
                            element = preprocess.encode(self._metadata, token_sequence, extended_sequence,
                                                        label_sequence, conll_txt, expanded_embedding=self._expanded_embedding)
                            if element is not None: self._input_queue.put(element)
            if not self._is_train:
                self._input_queue.put(None)

    def _FillBucketInputQueue(self):
        """Fill bucketed batches into the bucket_input_queue."""
        while True:
            inputs = []
            is_dataset_end = False
            for _ in range(self._batch_size * self._bucket_cache_batch):
                item = self._input_queue.get()
                if item is None:
                    is_dataset_end = True
                    break
                inputs.append(item)
            if self._is_train:
                inputs = sorted(inputs, key=lambda inp: inp.sequence_length)
            batches = preprocess.pad_and_batch(inputs, self._batch_size, self._metadata, self._is_train, expanded_embedding=self._expanded_embedding)

            if self._is_train:
                shuffle(batches)
            for b in batches:
                self._bucket_input_queue.put(b)
            if is_dataset_end:
                self._bucket_input_queue.put(None)

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
