# -*- coding: utf-8-*-
import time

import matplotlib
from tqdm import tqdm

import preprocess
from conlleval import evaluate_and_reports

matplotlib.use('Agg')
import sklearn
from tensorflow.contrib.tensorboard.plugins import projector

import evaluate
from data_queue import DataQueue
from params import BREAK_STEP
from evaluate import remap_labels

matplotlib.use('Agg')
import tensorflow as tf
import sklearn.linear_model
from entity_lstm import EntityLSTM
import os
import numpy as np
import utils
import pickle
import collections
from shutil import copyfile
import copy
from functools import partial

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class NeuroNER(object):
    def __init__(self, parameters, metadata):
        mode = parameters['mode']
        if mode == "predict" or mode == "vocab_expansion":
            parameters['batch_size'] = 1

        if mode == 'predict' and parameters['use_gazetteer']:
            gazetteer_path = parameters['pretrained_model_folder'] + '/gazetteer'
            pk = pickle.load(open(gazetteer_path, "rb"))
            self.gazetteer = pk['dic']
            self.max_key_len = pk['max_key_len']
            del pk
        else:
            self.gazetteer = None
            self.max_key_len = None

        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=parameters['number_of_cpu_threads'],
            inter_op_parallelism_threads=parameters['number_of_cpu_threads'],
            device_count={'CPU': 1, 'GPU': parameters['number_of_gpus']},
            allow_soft_placement=True,
            log_device_placement=False
        )

        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.90
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            if mode == "train" or mode == "vocab_expansion" or not parameters['use_vocab_expansion']:
                use_vocab_expansion = False
            else:
                use_vocab_expansion = True
            model = EntityLSTM(parameters, metadata, use_external_embedding=use_vocab_expansion)
            sess.run(tf.global_variables_initializer())

        if mode == "train":
            if parameters['load_pretrained_model']:
                self.transition_params_trained = model.load_model(parameters['pretrained_model_folder'], sess, metadata,
                                                                  parameters)
            else:
                model.load_pretrained_token_embeddings(sess, parameters, metadata)
                self.transition_params_trained = np.random.rand(metadata['num_of_label'] + 2,
                                                                metadata['num_of_label'] + 2)
        else:
            self.transition_params_trained = model.load_model(parameters['pretrained_model_folder'], sess, metadata,
                                                              parameters)

        expanded_embedding = None
        if parameters['use_vocab_expansion'] and (parameters['mode'] == 'predict' or parameters['mode'] == 'test'):
            expanded_embedding_filepath = parameters['pretrained_model_folder'] + "/expanded_embedding.pickles"
            if not os.path.exists(expanded_embedding_filepath):
                raise Exception("expand embedding file not exist")
            with open(expanded_embedding_filepath, "rb") as f:
                expanded_embedding = pickle.load(f)

        self.expanded_embedding = expanded_embedding
        self.model = model
        self.parameters = parameters
        self.sess = sess
        self.metadata = metadata

    def test(self, dataset_filepaths):
        stats_graph_folder, experiment_timestamp = self._create_stats_graph_folder(self.parameters)
        # Initialize and save execution details
        results = {}
        results['epoch'] = {}
        results['execution_details'] = {}
        results['execution_details']['train_start'] = 0
        results['execution_details']['time_stamp'] = experiment_timestamp
        results['execution_details']['early_stop'] = False
        results['execution_details']['keyboard_interrupt'] = False
        results['execution_details']['num_epochs'] = 0
        results['model_options'] = copy.copy(self.parameters)

        data_queue = {}
        for dataset_type, dataset_path in dataset_filepaths.items():
            data_queue[dataset_type] = DataQueue(self.metadata, dataset_path, self.parameters['batch_size'],
                                                 is_train=True if dataset_type == 'train' else False,
                                                 # use_process=True if dataset_type == 'train' else False,
                                                 use_process=False,
                                                 expanded_embedding=self.expanded_embedding,
                                                 pad_constant_size=self.parameters['use_attention'])
        epoch_start_time = time.time()
        accum_step = 0

        # Predict labels using trained model
        y_pred = {}
        y_true = {}
        output_filepaths = {}
        # for dataset_type in ['train', 'valid', 'test', 'deploy']:
        for dataset_type in ['valid', 'test', 'deploy']:
            if dataset_type not in data_queue:
                continue
            prediction_output = self._prediction_step(data_queue, dataset_type,
                                                      stats_graph_folder, accum_step)
            y_pred[dataset_type], y_true[dataset_type], output_filepaths[dataset_type] = prediction_output

        # Evaluate model: save and plot results
        evaluate.evaluate_model(results, self.metadata, y_pred, y_true, stats_graph_folder, accum_step,
                                epoch_start_time, output_filepaths, self.parameters)

    def predict(self, input):
        token_sequence, raw_token_sequence, extended_sequence = preprocess.extract_feature(input,
                                                                                           self.parameters['tokenizer'],
                                                                                           self.gazetteer,
                                                                                           self.max_key_len)
        model_input = preprocess.encode(self.metadata, token_sequence, extended_sequence,
                                        expanded_embedding=self.expanded_embedding)
        batch_input = preprocess.pad_and_batch([model_input], 1, self.metadata, is_train=False,
                                               expanded_embedding=self.expanded_embedding,
                                               pad_constant_size=self.parameters['use_attention'])
        _, prediction_labels_list, decode_score_list = self._predict_core(batch_input[0])
        return raw_token_sequence, extended_sequence, prediction_labels_list[0], decode_score_list[0]

    @staticmethod
    def batch_extract_feature(input, parameters, gazetteer, max_key_len, metadata, expanded_embedding):
        token_sequence, raw_token_sequence, extended_sequence = preprocess.extract_feature(input,
                                                                                           parameters[
                                                                                               'tokenizer'],
                                                                                           gazetteer,
                                                                                           max_key_len)
        model_input = preprocess.encode(metadata, token_sequence, extended_sequence,
                                        expanded_embedding=expanded_embedding)
        return raw_token_sequence, extended_sequence, model_input, raw_token_sequence

    def predict_list(self, input_list, pool):
        model_inputs = []
        raw_token_sequence_list = []
        extended_sequence_list = []

        partial_parse_token = partial(NeuroNER.batch_extract_feature, parameters=self.parameters, gazetteer=self.gazetteer,
                                      max_key_len=self.max_key_len, metadata=self.metadata,
                                      expanded_embedding=self.expanded_embedding)
        result = pool.map(partial_parse_token, input_list)

        for raw_token_sequence, extended_sequence, model_input, raw_token_sequence in result:
            # raw_token_sequence, extended_sequence, model_input, raw_token_sequence = result.next()
            model_inputs.append(model_input)
            raw_token_sequence_list.append(raw_token_sequence)
            extended_sequence_list.append(extended_sequence)

        batch_input_list = preprocess.pad_and_batch(model_inputs, self.parameters['batch_size'], self.metadata,
                                                    is_train=False,
                                                    expanded_embedding=self.expanded_embedding,
                                                    pad_constant_size=self.parameters['use_attention'])
        prediction_labels_list = []
        decode_score_list = []
        for batch_input in batch_input_list:
            _, current_prediction_labels_list, current_decode_score_list = self._predict_core(batch_input)
            prediction_labels_list.extend(current_prediction_labels_list)
            decode_score_list.extend(current_decode_score_list)
        return raw_token_sequence_list, extended_sequence_list, prediction_labels_list, decode_score_list

    def fit(self, dataset_filepaths):
        stats_graph_folder, experiment_timestamp = self._create_stats_graph_folder(self.parameters)
        # Initialize and save execution details
        start_time = time.time()
        results = {}
        results['epoch'] = {}
        results['execution_details'] = {}
        results['execution_details']['train_start'] = start_time
        results['execution_details']['time_stamp'] = experiment_timestamp
        results['execution_details']['early_stop'] = False
        results['execution_details']['keyboard_interrupt'] = False
        results['execution_details']['num_epochs'] = 0
        results['model_options'] = copy.copy(self.parameters)

        model_folder = os.path.join(stats_graph_folder, 'model')
        utils.create_folder_if_not_exists(model_folder)
        del self.metadata['prev_num_of_token']
        del self.metadata['prev_num_of_char']

        copyfile(self.parameters['ini_path'], os.path.join(model_folder, 'parameters.ini'))

        if self.parameters['enable_tensorbord']:
            tensorboard_log_folder = os.path.join(stats_graph_folder, 'tensorboard_logs')
            utils.create_folder_if_not_exists(tensorboard_log_folder)
            tensorboard_log_folders = {}
            for dataset_type in dataset_filepaths.keys():
                tensorboard_log_folders[dataset_type] = os.path.join(stats_graph_folder, 'tensorboard_logs',
                                                                     dataset_type)
                utils.create_folder_if_not_exists(tensorboard_log_folders[dataset_type])

            # Instantiate the writers for TensorBoard
            writers = {}
            for dataset_type in dataset_filepaths.keys():
                writers[dataset_type] = tf.summary.FileWriter(tensorboard_log_folders[dataset_type],
                                                              graph=self.sess.graph)
            embedding_writer = tf.summary.FileWriter(
                model_folder)  # embedding_writer has to write in model_folder, otherwise TensorBoard won't be able to view embeddings

            embeddings_projector_config = projector.ProjectorConfig()
            tensorboard_token_embeddings = embeddings_projector_config.embeddings.add()
            tensorboard_token_embeddings.tensor_name = self.model.token_embedding_weights.name
            token_list_file_path = os.path.join(model_folder, 'tensorboard_metadata_tokens.tsv')
            tensorboard_token_embeddings.metadata_path = os.path.relpath(token_list_file_path, '..')
            if self.parameters['use_character_lstm']:
                tensorboard_character_embeddings = embeddings_projector_config.embeddings.add()
                tensorboard_character_embeddings.tensor_name = self.model.character_embedding_weights.name
                character_list_file_path = os.path.join(model_folder, 'tensorboard_metadata_characters.tsv')
                tensorboard_character_embeddings.metadata_path = os.path.relpath(character_list_file_path, '..')

            projector.visualize_embeddings(embedding_writer, embeddings_projector_config)

            # Write metadata for TensorBoard embeddings
            token_list_file = open(token_list_file_path, 'w', encoding='UTF-8')
            for key, _ in self.metadata['token_to_index'].items():
                token_list_file.write('{0}\n'.format(key))
            token_list_file.close()

            if self.parameters['use_character_lstm']:
                character_list_file = open(character_list_file_path, 'w', encoding='UTF-8')
                for key, _ in self.metadata['character_to_index'].items():
                    character_list_file.write('{0}\n'.format(key))
                character_list_file.close()

        # Start training + evaluation loop. Each iteration corresponds to 1 epoch.
        bad_counter = 0  # number of epochs with no improvement on the validation test in terms of F1-score
        previous_best_valid_f1_score = 0

        data_queue = {}
        for dataset_type, dataset_path in dataset_filepaths.items():
            data_queue[dataset_type] = DataQueue(self.metadata, dataset_path, self.parameters['batch_size'],
                                                 is_train=True if dataset_type == 'train' else False,
                                                 # use_process=True if dataset_type == 'train' else False,
                                                 use_process=False,
                                                 pad_constant_size=self.parameters['use_attention'])

        first_step = True
        try:
            accum_step = 0
            step = 0
            while True:
                print('\nStarting step {0}'.format(accum_step))

                epoch_start_time = time.time()

                if not first_step:
                    bar = tqdm(total=BREAK_STEP)
                    while True:
                        if step > BREAK_STEP:
                            step %= BREAK_STEP
                            break
                        batch_input = data_queue['train'].next()
                        self.transition_params_trained = self._train_step(batch_input)
                        step += self.parameters['batch_size']
                        accum_step += self.parameters['batch_size']
                        # print('Training {0:.2f}% done'.format(step / BREAK_STEP * 100), end='\r', flush=True)
                        bar.update(self.parameters['batch_size'])
                    epoch_elapsed_training_time = time.time() - epoch_start_time
                    print('Training completed in {0:.2f} seconds'.format(epoch_elapsed_training_time), flush=True)
                    bar.close()
                first_step = False

                # Predict labels using trained model
                y_pred = {}
                y_true = {}
                output_filepaths = {}
                # for dataset_type in ['train', 'valid', 'test', 'deploy']:
                for dataset_type in ['valid', 'test', 'deploy']:
                    if dataset_type not in data_queue:
                        continue
                    prediction_output = self._prediction_step(data_queue, dataset_type,
                                                              stats_graph_folder, accum_step)
                    y_pred[dataset_type], y_true[dataset_type], output_filepaths[dataset_type] = prediction_output

                # Evaluate model: save and plot results
                evaluate.evaluate_model(results, self.metadata, y_pred, y_true, stats_graph_folder, accum_step,
                                        epoch_start_time, output_filepaths, self.parameters)

                # Save model
                self.model.saver.save(self.sess, os.path.join(model_folder, 'model_{0:05d}.ckpt'.format(accum_step)))
                self.metadata.write(model_folder)

                if self.parameters['enable_tensorbord']:
                    # Save TensorBoard logs
                    summary = self.sess.run(self.model.summary_op, feed_dict=None)
                    writers['train'].add_summary(summary, accum_step)
                    writers['train'].flush()
                    utils.copytree(writers['train'].get_logdir(), model_folder)

                # Early stop
                valid_f1_score = results['epoch'][accum_step][0]['valid']['f1_score']['micro']
                if valid_f1_score > previous_best_valid_f1_score:
                    bad_counter = 0
                    previous_best_valid_f1_score = valid_f1_score
                    ##
                    self.model.saver.save(self.sess,
                                          os.path.join(r'D:\tech\entity\NeuroNER\trained_models\exobrain',
                                                       'model.ckpt'))
                    self.metadata.write(r'D:\tech\entity\NeuroNER\trained_models\exobrain')
                    ##
                else:
                    bad_counter += 1
                print("The last {0} epochs have not shown improvements on the validation set.".format(bad_counter))

                if bad_counter >= self.parameters['patience']:
                    print('Early Stop!')
                    results['execution_details']['early_stop'] = True
                    break

                if accum_step >= self.parameters['maximum_number_of_steps']: break


        except KeyboardInterrupt:
            results['execution_details']['keyboard_interrupt'] = True
            print('Training interrupted')

        print('Finishing the experiment')

        end_time = time.time()
        results['execution_details']['train_duration'] = end_time - start_time
        results['execution_details']['train_end'] = end_time
        evaluate.save_results(results, stats_graph_folder)
        if self.parameters['enable_tensorbord']:
            for dataset_type in dataset_filepaths.keys():
                writers[dataset_type].close()

    def vocab_expansion(self):
        trained_emb = self.sess.run(self.model.token_embedding_weights.read_value())
        word2vec = utils.load_pretrained_token_embeddings(self.parameters['token_pretrained_embedding_filepath'])

        # Find words shared between the two vocabularies.
        print("Finding shared words")
        shared_words = [w for w in word2vec.keys() if w in self.metadata['token_to_index']]

        # Select embedding vectors for shared words.
        print("Selecting embeddings for %d shared words", len(shared_words))
        shared_st_emb = trained_emb[[self.metadata['token_to_index'][w] for w in shared_words]]
        shared_w2v_emb = [word2vec[w] for w in shared_words]

        # Train a linear regression model on the shared embedding vectors.
        print("Training linear regression model")
        model = sklearn.linear_model.LinearRegression()
        model.fit(shared_w2v_emb, shared_st_emb)

        # Create the expanded vocabulary.
        print("Creating embeddings for expanded vocabuary")
        expanded_embedding = collections.OrderedDict()
        for w, v in word2vec.items():
            # Ignore words with underscores (spaces).
            if "_" not in w:
                w_emb = model.predict(v.reshape(1, -1))
                expanded_embedding[w] = w_emb.reshape(-1)

        for w, i in self.metadata['token_to_index'].items():
            expanded_embedding[w] = trained_emb[i]

        print("Created expanded vocabulary of %d words", len(expanded_embedding))

        # Save the output.
        expanded_embedding_filepath = self.parameters['pretrained_model_folder'] + "/expanded_embedding.pickles"
        pickle.dump(expanded_embedding, open(expanded_embedding_filepath, "wb"))

    def _create_stats_graph_folder(self, parameters):
        # Initialize stats_graph_folder
        experiment_timestamp = utils.get_current_time_in_miliseconds()
        dataset_name = utils.get_basename_without_extension(parameters['dataset_text_folder'])
        model_name = '{0}_{1}'.format(dataset_name, experiment_timestamp)
        utils.create_folder_if_not_exists(parameters['output_folder'])
        stats_graph_folder = os.path.join(parameters['output_folder'], model_name)  # Folder where to save graphs
        utils.create_folder_if_not_exists(stats_graph_folder)
        return stats_graph_folder, experiment_timestamp

    def _train_step(self, batch_input):
        feed_dict = {
            self.model.input_token_indices: batch_input['token_indices'],
            self.model.input_extended_sequence: batch_input['extended_sequence'],
            self.model.input_token_character_indices: batch_input['character_indices'],
            self.model.input_token_lengths: batch_input['token_lengths'],
            self.model.input_label_indices_flat: batch_input['label_indices'],
            self.model.input_seq_lengths: batch_input['seq_lengths'],
            self.model.training: True,
            self.model.dropout_keep_prob: 1 - self.parameters['dropout_rate']
        }
        _, _, loss, accuracy, transition_params_trained = self.sess.run(
            [self.model.train_op, self.model.global_step, self.model.loss, self.model.accuracy,
             self.model.transition_parameters], feed_dict)
        return transition_params_trained

    def _predict_core(self, batch_input):
        feed_dict = {
            self.model.input_token_indices: batch_input['token_indices'],
            self.model.input_extended_sequence: batch_input['extended_sequence'],
            self.model.input_token_character_indices: batch_input['character_indices'],
            self.model.input_token_lengths: batch_input['token_lengths'],
            self.model.input_seq_lengths: batch_input['seq_lengths'],
            self.model.training: False,
            self.model.dropout_keep_prob: 1.
        }
        batch_unary_scores, batch_predictions = self.sess.run([self.model.unary_scores, self.model.predictions],
                                                              feed_dict)

        predictions_list = []
        prediction_labels_list = []
        decode_score_list = []
        for idx in range(batch_input['batch_size']):
            unary_scores = batch_unary_scores[idx]

            unary_scores = unary_scores[:batch_input['seq_lengths'][idx] + 2]
            unary_scores[:][batch_input['seq_lengths'][idx] + 1] = -1000
            unary_scores[-1][-1] = 0
            predictions = batch_predictions[idx]
            if self.parameters['use_crf']:
                predictions, decode_score = tf.contrib.crf.viterbi_decode(unary_scores, self.transition_params_trained)
                predictions = predictions[1:batch_input['seq_lengths'][idx] + 1]
            else:
                predictions = predictions[:batch_input['seq_lengths'][idx]].tolist()

            decode_score_list.append(decode_score)
            predictions_list.append(predictions)
            prediction_labels_list.append([self.metadata['index_to_label'][prediction] for prediction in predictions])
        return predictions_list, prediction_labels_list, decode_score_list

    def _prediction_step(self, data_queue, dataset_type, stats_graph_folder, train_step):
        all_predictions = []
        all_y_true = []
        output_filepath = os.path.join(stats_graph_folder, '{1:03d}_{0}.txt'.format(dataset_type, train_step))
        output_file = open(output_filepath, 'w', encoding='UTF-8')
        step = 0

        while True:
            batch_input = data_queue[dataset_type].next()
            if batch_input is None:
                break
            predictions_list, prediction_labels_list, _ = self._predict_core(batch_input)
            for idx in range(batch_input['batch_size']):
                predictions = predictions_list[idx]
                prediction_labels = prediction_labels_list[idx]
                if dataset_type == 'deploy':
                    conll_text = batch_input['conll'][idx].split('\n')
                    output_string = ""
                    for line, prediction in zip(conll_text, prediction_labels):
                        split_line = line.strip().split(' ')
                        split_line.append(prediction)
                        output_string += ' '.join(split_line) + '\n'
                    output_file.write(output_string + '\n')
                else:
                    gold_labels = [self.metadata['index_to_label'][t] for t in
                                   batch_input['label_indices'][idx][:batch_input['seq_lengths'][idx]]]

                    conll_text = batch_input['conll'][idx].split('\n')
                    output_string = ""
                    for line, prediction, gold_label in zip(conll_text, prediction_labels, gold_labels):
                        split_line = line.strip().split(' ')
                        gold_label_original = split_line[-1]
                        assert (gold_label == gold_label_original)
                        split_line.append(prediction)
                        output_string += ' '.join(split_line) + '\n'
                    output_file.write(output_string + '\n')
                    all_y_true.extend(batch_input['label_indices'][idx][:batch_input['seq_lengths'][idx]])

                all_predictions.extend(predictions)

            step += self.parameters['batch_size']

        output_file.close()

        if dataset_type != 'deploy':
            if self.parameters['main_evaluation_mode'] == 'conll':
                conll_output_filepath = '{0}_conll_evaluation.txt'.format(output_filepath)
                evaluate_and_reports(output_filepath, conll_output_filepath)
                with open(conll_output_filepath, 'r') as f:
                    classification_report = f.read()
                    print(classification_report)
            else:
                new_y_pred, new_y_true, new_label_indices, new_label_names, _, _ = remap_labels(all_predictions,
                                                                                                all_y_true,
                                                                                                self.metadata,
                                                                                                self.parameters[
                                                                                                    'main_evaluation_mode'])
                print(sklearn.metrics.classification_report(new_y_true, new_y_pred, digits=4, labels=new_label_indices,
                                                            target_names=new_label_names))

        return all_predictions, all_y_true, output_filepath

    def close(self):
        self.__del__()

    def __del__(self):
        self.sess.close()
