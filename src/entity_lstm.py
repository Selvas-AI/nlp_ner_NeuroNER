# -*- coding: utf-8-*-
import os
import re
import time

import tensorflow as tf

import bnlstm
import utils
from attention import attention_decoder
from params import LIMIT_SEQUENCE_LENGTH


def bidirectional_LSTM(input, training, hidden_state_dimension, initializer, sequence_length=None,
                       output_sequence=True, lstm_cell_type='lstm'):
    with tf.variable_scope("bidirectional_LSTM"):
        if sequence_length == None:
            batch_size = 1
            sequence_length = tf.shape(input)[1]
            sequence_length = tf.expand_dims(sequence_length, axis=0, name='sequence_length')
        else:
            batch_size = tf.shape(sequence_length)[0]

        lstm_cell = {}
        initial_state = {}
        for direction in ["forward", "backward"]:
            with tf.variable_scope(direction):
                # LSTM cell
                if lstm_cell_type == 'bnlstm':
                    lstm_cell[direction] = bnlstm.BN_LSTMCell(hidden_state_dimension,
                                                              training,
                                                              forget_bias=1.0,
                                                              initializer=initializer,
                                                              state_is_tuple=True)
                elif lstm_cell_type == 'lnlstm':
                    lstm_cell[direction] = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_state_dimension,
                                                                                 forget_bias=1.0, )
                else:
                    lstm_cell[direction] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(hidden_state_dimension,
                                                                                         forget_bias=1.0,
                                                                                         initializer=initializer,
                                                                                         state_is_tuple=True)

                # initial state: http://stackoverflow.com/questions/38441589/tensorflow-rnn-initial-state
                initial_cell_state = tf.get_variable("initial_cell_state", shape=[1, hidden_state_dimension],
                                                     dtype=tf.float32, initializer=initializer)
                initial_output_state = tf.get_variable("initial_output_state", shape=[1, hidden_state_dimension],
                                                       dtype=tf.float32, initializer=initializer)
                c_states = tf.tile(initial_cell_state, tf.stack([batch_size, 1]))
                h_states = tf.tile(initial_output_state, tf.stack([batch_size, 1]))
                initial_state[direction] = tf.contrib.rnn.LSTMStateTuple(c_states, h_states)

        # sequence_length must be provided for tf.nn.bidirectional_dynamic_rnn due to internal bug
        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell["forward"],
                                                                lstm_cell["backward"],
                                                                input,
                                                                dtype=tf.float32,
                                                                sequence_length=sequence_length,
                                                                initial_state_fw=initial_state["forward"],
                                                                initial_state_bw=initial_state["backward"])
        if output_sequence == True:
            outputs_forward, outputs_backward = outputs
            output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
        else:
            # max pooling
            #             outputs_forward, outputs_backward = outputs
            #             output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
            #             output = tf.reduce_max(output, axis=1, name='output')
            # last pooling
            final_states_forward, final_states_backward = final_states
            output = tf.concat([final_states_forward[1], final_states_backward[1]], axis=1, name='output')

    return output, final_states


def MultiBlstm(input, input_seq_lengths, dropout_keep_prob, training, initializer, parameters):
    # Add dropout
    if type(parameters['token_lstm_hidden_state_dimension']) is not list:
        hidden_state_dims = [parameters['token_lstm_hidden_state_dimension']]
    else:
        hidden_state_dims = parameters['token_lstm_hidden_state_dimension']

    for index, hd in enumerate(hidden_state_dims):
        with tf.variable_scope("MultiBlstm_%d" % index):
            with tf.variable_scope("dropout"):
                if 'token_lstm_output' in locals():
                    input = token_lstm_output
                token_lstm_input_drop = tf.nn.dropout(input, dropout_keep_prob, name='token_lstm_input_drop')
                '''
                if self.verbose: print("token_lstm_input_drop: {0}".format(token_lstm_input_drop))
                # https://www.tensorflow.org/api_guides/python/contrib.rnn
                # Prepare data shape to match `rnn` function requirements
                # Current data input shape: (batch_size, n_steps, n_input)
                # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
                '''

            # Token LSTM layer
            with tf.variable_scope('token_lstm') as vs:
                token_lstm_output, final_states = bidirectional_LSTM(token_lstm_input_drop,
                                                                     training,
                                                                     hd, initializer,
                                                                     sequence_length=input_seq_lengths,
                                                                     output_sequence=True,
                                                                     lstm_cell_type=parameters[
                                                                         'lstm_cell_type'])
                if 'token_lstm_variables' in locals():
                    token_lstm_variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
                else:
                    token_lstm_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
    return token_lstm_output, hd, token_lstm_variables, final_states


def variable_summaries(var):
    '''
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    From https://www.tensorflow.org/get_started/summaries_and_tensorboard
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def resize_tensor_variable(sess, tensor_variable, shape):
    sess.run(tf.assign(tensor_variable, tf.zeros(shape), validate_shape=False))


class EntityLSTM(object):
    """
    An LSTM architecture for named entity recognition.
    Uses a character embedding layer followed by an LSTM to generate vector representation from characters for each token.
    Then the character vector is concatenated with token embedding vector, which is input to another LSTM  followed by a CRF layer.
    """

    def __init__(self, parameters, metadata, use_external_embedding=False):
        num_of_char = metadata['num_of_char']
        num_of_token = metadata['num_of_token']
        num_of_label = metadata['num_of_label']
        num_of_extendeds = metadata['num_of_extendeds']

        self.verbose = False
        self.batch_size = parameters['batch_size']
        # Placeholders for input, output and dropout
        if use_external_embedding:
            self.input_token_indices = tf.placeholder(tf.float32,
                                                      (self.batch_size, None, parameters['token_embedding_dimension']),
                                                      "input_token_indices")
        else:
            self.input_token_indices = tf.placeholder(tf.int32, [self.batch_size, None], name="input_token_indices")
        self.input_extended_sequence = tf.placeholder(tf.int32, [self.batch_size, None, None],
                                                      name="input_token_space_indices")
        self.input_label_indices_flat = tf.placeholder(tf.int32, [self.batch_size, None],
                                                       name="input_label_indices_flat")
        self.input_token_character_indices = tf.placeholder(tf.int32, [self.batch_size, None, None],
                                                            name="input_token_character_indices")
        self.input_token_lengths = tf.placeholder(tf.int32, [self.batch_size, None], name="input_token_lengths")
        self.input_seq_lengths = tf.placeholder(tf.int32, [self.batch_size], name="input_seq_lengths")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.training = tf.placeholder(tf.bool, name="training")

        # Internal parameters
        initializer = tf.contrib.layers.xavier_initializer()

        itl_shape = tf.shape(self.input_token_lengths)
        if parameters['use_character_lstm']:
            # Character-level LSTM
            # Idea: reshape so that we have a tensor [number_of_token, max_token_length, token_embeddings_size], which we pass to the LSTM

            # Character embedding layer
            with tf.variable_scope("character_embedding"):
                self.character_embedding_weights = tf.get_variable(
                    "character_embedding_weights",
                    shape=[num_of_char, parameters['character_embedding_dimension']],
                    initializer=initializer)

                itci_shape = tf.shape(self.input_token_character_indices)
                reshaped_itci = tf.reshape(self.input_token_character_indices,
                                           [itci_shape[0] * itci_shape[1], itci_shape[2]])
                embedded_characters = tf.nn.embedding_lookup(self.character_embedding_weights,
                                                             reshaped_itci,
                                                             name='embedded_characters')
                if self.verbose: print("embedded_characters: {0}".format(embedded_characters))
                variable_summaries(self.character_embedding_weights)

            # Character LSTM layer
            with tf.variable_scope('character_lstm') as vs:
                reshaped_itl = tf.reshape(self.input_token_lengths,
                                          [itl_shape[0] * itl_shape[1]])
                character_lstm_output, _ = bidirectional_LSTM(embedded_characters,
                                                              self.training,
                                                              parameters['character_lstm_hidden_state_dimension'],
                                                              initializer,
                                                              sequence_length=reshaped_itl,
                                                              output_sequence=False)
                reshaped_character_lstm_output = tf.reshape(character_lstm_output,
                                                            [itl_shape[0], itl_shape[1], parameters[
                                                                'character_lstm_hidden_state_dimension'] * 2])
                self.character_lstm_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        # Token embedding layer
        with tf.variable_scope("token_embedding"):
            if use_external_embedding:
                embedded_tokens = self.input_token_indices
            else:
                self.token_embedding_weights = tf.get_variable(
                    "token_embedding_weights",
                    shape=[num_of_token, parameters['token_embedding_dimension']],
                    initializer=initializer,
                    trainable=not parameters['freeze_token_embeddings'])
                iti_shape = tf.shape(self.input_token_indices)
                reshaped_iti = tf.reshape(self.input_token_indices, [iti_shape[0] * iti_shape[1]])
                embedded_tokens = tf.nn.embedding_lookup(self.token_embedding_weights, reshaped_iti)

            stacked_embedded_tokens = tf.reshape(embedded_tokens,
                                                 [itl_shape[0], itl_shape[1], parameters['token_embedding_dimension']])
            if not use_external_embedding:
                variable_summaries(self.token_embedding_weights)

        ext_shape = tf.shape(self.input_extended_sequence)
        splitted_extended_value = tf.split(self.input_extended_sequence, num_or_size_splits=len(num_of_extendeds),
                                           axis=2)
        for ex_idx, extended in enumerate(splitted_extended_value):
            with tf.variable_scope("extended_%d" % ex_idx):
                reshaped_extended = tf.reshape(extended, [ext_shape[0], ext_shape[1]])
                reshaped_extended_oh = tf.one_hot(reshaped_extended, num_of_extendeds[ex_idx])
                stacked_embedded_tokens = tf.concat([stacked_embedded_tokens, reshaped_extended_oh], axis=2)

        # Concatenate character LSTM outputs and token embeddings
        if parameters['use_character_lstm']:
            with tf.variable_scope("concatenate_token_and_character_vectors"):
                if self.verbose: print('embedded_tokens: {0}'.format(stacked_embedded_tokens))
                token_lstm_input = tf.concat(
                    [reshaped_character_lstm_output, stacked_embedded_tokens], axis=2,
                    name='token_lstm_input')
                if self.verbose: print("token_lstm_input: {0}".format(token_lstm_input))
        else:
            token_lstm_input = stacked_embedded_tokens

        token_lstm_output, hidden_dim, self.token_lstm_variables, final_states = MultiBlstm(token_lstm_input,
                                                                                            self.input_seq_lengths,
                                                                                            self.dropout_keep_prob,
                                                                                            self.training, initializer,
                                                                                            parameters)
        if parameters['use_attention']:
            # state = final_states[-1]
            # encoder_state = tf.concat(state, 1)
            encoder_state_fw, encoder_state_bw = final_states
            state_fw = encoder_state_fw[-1]
            state_bw = encoder_state_bw[-1]
            encoder_state = tf.concat([tf.concat(state_fw, 1),
                                       tf.concat(state_bw, 1)], 1)
            encoder_out, self.attention_weight = attention_decoder(token_lstm_output,
                                                                   encoder_state,
                                                                   parameters['attention_size'],
                                                                   LIMIT_SEQUENCE_LENGTH)
        else:
            encoder_out = token_lstm_output

        # Needed only if Bidirectional LSTM is used for token level
        with tf.variable_scope("feedforward_after_lstm") as vs:
            shape_tlo = tf.shape(encoder_out)
            if parameters['use_attention']:
                prev_node_size = parameters['attention_size']
            else:
                prev_node_size = hidden_dim * 2

            W = tf.get_variable(
                "W",
                shape=[prev_node_size, hidden_dim],
                initializer=initializer)
            b = tf.Variable(tf.constant(0.0, shape=[hidden_dim]), name="bias")
            reshaped_tlo = tf.reshape(encoder_out, [shape_tlo[0] * shape_tlo[1], shape_tlo[2]])
            outputs = tf.nn.xw_plus_b(reshaped_tlo, W, b, name="output_before_tanh")
            outputs = tf.nn.tanh(outputs, name="output_after_tanh")
            variable_summaries(W)
            variable_summaries(b)
            self.token_lstm_variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope("feedforward_before_crf") as vs:
            W = tf.get_variable(
                "W",
                shape=[hidden_dim, num_of_label],
                initializer=initializer)
            b = tf.Variable(tf.constant(0.0, shape=[num_of_label]), name="bias")
            scores = tf.nn.xw_plus_b(outputs, W, b, name="scores")
            shape_scores = tf.shape(scores)
            self.unary_scores = tf.reshape(scores, [self.batch_size, tf.div(shape_scores[0], self.batch_size),
                                                    num_of_label])
            self.predictions = tf.argmax(self.unary_scores, 2, name="predictions")
            variable_summaries(W)
            variable_summaries(b)
            self.feedforward_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        # CRF layer
        if parameters['use_crf']:
            with tf.variable_scope("crf") as vs:
                # Add start and end tokens
                small_score = -1000.0
                large_score = 0.0

                unary_scores_with_start_and_end = tf.concat(
                    [self.unary_scores, tf.tile(tf.constant(small_score, shape=[1, 1, 2]),
                                                [tf.shape(self.unary_scores)[0], tf.shape(self.unary_scores)[1], 1])],
                    2)

                start_unary_scores = [[[small_score] * num_of_label + [large_score,
                                                                       small_score]]] * self.batch_size
                end_unary_scores = [[[small_score] * num_of_label + [small_score,
                                                                     large_score]]] * self.batch_size
                self.unary_scores = tf.concat([start_unary_scores, unary_scores_with_start_and_end, end_unary_scores],
                                              1)


                start_index = num_of_label
                end_index = num_of_label + 1
                input_label_indices_flat_with_start_and_end = tf.concat(
                    [tf.constant(start_index, shape=[self.batch_size, 1]), self.input_label_indices_flat,
                     tf.constant(end_index, shape=[self.batch_size, 1])], 1)

                # Apply CRF layer
                if self.verbose: print('unary_scores_expanded: {0}'.format(self.unary_scores))
                if self.verbose: print(
                    'input_label_indices_flat_batch: {0}'.format(input_label_indices_flat_with_start_and_end))
                if self.verbose: print("sequence_lengths: {0}".format(self.input_seq_lengths))
                # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/crf
                # Compute the log-likelihood of the gold sequences and keep the transition params for inference at test time.
                self.transition_parameters = tf.get_variable(
                    "transitions",
                    shape=[num_of_label + 2, num_of_label + 2],
                    initializer=initializer)
                variable_summaries(self.transition_parameters)
                log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                    self.unary_scores, input_label_indices_flat_with_start_and_end, self.input_seq_lengths,
                    transition_params=self.transition_parameters)
                self.loss = tf.reduce_mean(-log_likelihood, name='cross_entropy_mean_loss')
                self.accuracy = tf.constant(1)

                self.crf_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        # Do not use CRF layer
        else:
            with tf.variable_scope("crf") as vs:
                self.transition_parameters = tf.get_variable(
                    "transitions",
                    shape=[num_of_label + 2, num_of_label + 2],
                    initializer=initializer)
                variable_summaries(self.transition_parameters)
                self.crf_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

            # Calculate mean cross-entropy loss
            input_label_indices_vector_oh = tf.one_hot(self.input_label_indices_vector, num_of_label)
            with tf.variable_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.unary_scores,
                                                                 labels=input_label_indices_vector_oh, name='softmax')
                self.loss = tf.reduce_mean(losses, name='cross_entropy_mean_loss')
            with tf.variable_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(input_label_indices_vector_oh, 2))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

        if parameters['mode'] == 'train':
            self.define_training_procedure(parameters)
            self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=parameters['patience'] + 1)  # defaults to saving all variables

    def define_training_procedure(self, parameters):
        # Define training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        if parameters['optimizer'] == 'adam':
            self.optimizer = tf.train.AdamOptimizer(parameters['learning_rate'])
        elif parameters['optimizer'] == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(parameters['learning_rate'])
        elif parameters['optimizer'] == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(parameters['learning_rate'])
        else:
            raise ValueError('The lr_method parameter must be either adadelta, adam or sgd.')

        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        if parameters['gradient_clipping_value']:
            grads_and_vars = [(tf.clip_by_value(grad, -parameters['gradient_clipping_value'],
                                                parameters['gradient_clipping_value']), var)
                              for grad, var in grads_and_vars]
        # By defining a global_step variable and passing it to the optimizer we allow TensorFlow handle the counting of training steps for us.
        # The global step will be automatically incremented by one every time you execute train_op.
        self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def load_pretrained_token_embeddings(self, sess, parameters, metadata):
        if parameters['token_pretrained_embedding_filepath'] == '':
            return
        if not os.path.exists(parameters['token_pretrained_embedding_filepath']):
            raise Exception("Embedding file not exist")
        # Load embeddings
        start_time = time.time()
        print('Load token embeddings... ', end='', flush=True)
        token_to_vector = utils.load_pretrained_token_embeddings(parameters['token_pretrained_embedding_filepath'])

        initial_weights = sess.run(self.token_embedding_weights.read_value())
        number_of_loaded_word_vectors = 0
        number_of_token_original_case_found = 0
        number_of_token_lowercase_found = 0
        number_of_token_digits_replaced_with_zeros_found = 0
        number_of_token_lowercase_and_digits_replaced_with_zeros_found = 0
        for token in metadata['token_to_index'].keys():
            if token in token_to_vector.keys():
                initial_weights[metadata['token_to_index'][token]] = token_to_vector[token]
                number_of_token_original_case_found += 1
            elif parameters['check_for_lowercase'] and token.lower() in token_to_vector.keys():
                initial_weights[metadata['token_to_index'][token]] = token_to_vector[token.lower()]
                number_of_token_lowercase_found += 1
            elif parameters['check_for_digits_replaced_with_zeros'] and re.sub('\d', '0',
                                                                               token) in token_to_vector.keys():
                initial_weights[metadata['token_to_index'][token]] = token_to_vector[re.sub('\d', '0', token)]
                number_of_token_digits_replaced_with_zeros_found += 1
            elif parameters['check_for_lowercase'] and parameters['check_for_digits_replaced_with_zeros'] and re.sub(
                    '\d', '0', token.lower()) in token_to_vector.keys():
                initial_weights[metadata['token_to_index'][token]] = token_to_vector[re.sub('\d', '0', token.lower())]
                number_of_token_lowercase_and_digits_replaced_with_zeros_found += 1
            else:
                continue
            number_of_loaded_word_vectors += 1
        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))
        print("number_of_token_original_case_found: {0}".format(number_of_token_original_case_found))
        print("number_of_token_lowercase_found: {0}".format(number_of_token_lowercase_found))
        print("number_of_token_digits_replaced_with_zeros_found: {0}".format(
            number_of_token_digits_replaced_with_zeros_found))
        print("number_of_token_lowercase_and_digits_replaced_with_zeros_found: {0}".format(
            number_of_token_lowercase_and_digits_replaced_with_zeros_found))
        print('number_of_loaded_word_vectors: {0}'.format(number_of_loaded_word_vectors))
        print("dataset.vocabulary_size: {0}".format(metadata['num_of_token']))
        sess.run(self.token_embedding_weights.assign(initial_weights))

    def load_embeddings_from_pretrained_model(self, sess, pretrained_embedding_weights, embedding_type='token'):
        if embedding_type == 'token':
            embedding_weights = self.token_embedding_weights
        elif embedding_type == 'character':
            embedding_weights = self.character_embedding_weights

        start_time = time.time()
        print('Load {0} embeddings from pretrained model... '.format(embedding_type), end='', flush=True)
        initial_weights = sess.run(embedding_weights.read_value())

        for index in range(len(pretrained_embedding_weights)):
            initial_weights[index] = pretrained_embedding_weights[index]
        sess.run(embedding_weights.assign(initial_weights))
        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))

    def load_model(self, pretrained_model_folder, sess, metadata, parameters):
        reload_char = metadata['prev_num_of_char'] != metadata['num_of_char']
        reload_token = metadata['prev_num_of_token'] != metadata['num_of_token']

        pretrained_model_checkpoint_filepath = os.path.join(pretrained_model_folder, 'model.ckpt')
        if not reload_char and not reload_token:
            self.saver.restore(sess, pretrained_model_checkpoint_filepath)
        else:
            var_list = tf.trainable_variables()
            for idx, tensor in reversed(list(enumerate(var_list))):
                if reload_char and tensor.name == 'character_embedding/character_embedding_weights:0':
                    del var_list[idx]
                if reload_token and tensor.name == 'token_embedding/token_embedding_weights:0':
                    del var_list[idx]
            embed_wo_saver = tf.train.Saver(var_list=var_list)
            embed_wo_saver.restore(sess, pretrained_model_checkpoint_filepath)

            init_var = []
            graph = tf.Graph()
            with graph.as_default():
                local_sess = tf.Session()
                with local_sess.as_default():
                    if reload_char:
                        with tf.variable_scope("character_embedding"):
                            char_embed = tf.get_variable(
                                "character_embedding_weights",
                                shape=[metadata['prev_num_of_char'], parameters['character_embedding_dimension']],
                                trainable=True)
                        init_var.append(self.character_embedding_weights)
                    if reload_token:
                        with tf.variable_scope("token_embedding"):
                            token_embed = tf.get_variable(
                                "token_embedding_weights",
                                shape=[metadata['prev_num_of_token'], parameters['token_embedding_dimension']],
                                trainable=True)
                        init_var.append(self.token_embedding_weights)
                    embed_saver = tf.train.Saver()
                    embed_saver.restore(local_sess, pretrained_model_checkpoint_filepath)
                    if reload_char:
                        character_embedding_weights = local_sess.run([char_embed])
                    if reload_token:
                        token_embedding_weights = local_sess.run([token_embed])

            sess.run(tf.variables_initializer(init_var))

            if reload_char:
                self.load_embeddings_from_pretrained_model(sess, character_embedding_weights[0],
                                                           embedding_type='character')
            if reload_token:
                self.load_pretrained_token_embeddings(sess, parameters, metadata)
                self.load_embeddings_from_pretrained_model(sess, token_embedding_weights[0], embedding_type='token')
            del character_embedding_weights
            del token_embedding_weights
            del local_sess
            del graph

        return sess.run(self.transition_parameters)