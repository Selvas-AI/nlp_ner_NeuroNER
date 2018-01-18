# -*- coding: utf-8 -*-
"""
Created on Sun Feb  28 11:32:21 2016

@author: Bing Liu (liubing@cmu.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl

linear = rnn_cell_impl._linear


# linear = tf.nn.rnn_cell._linear

def attention_decoder(encoder_outputs_raw,
                      encoder_state,
                      num_decoder_symbols,
                      timestep_size,
                      num_heads=1,
                      dtype=tf.float32,
                      scope=None):
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")

    with tf.variable_scope(scope or "attention_RNN"):
        encoder_outputs = array_ops.transpose(encoder_outputs_raw, [1, 0, 2])
        output_size = encoder_outputs.get_shape()[2].value
        top_states = [tf.reshape(encoder_outputs[ti], [-1, 1, output_size])
                      for ti in range(timestep_size)]
        attention_states = tf.concat(top_states, 1)
        if not attention_states.get_shape()[1:2].is_fully_defined():
            raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                             % attention_states.get_shape())

        batch_size = tf.shape(top_states[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = tf.reshape(
            attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in range(num_heads):
            k = tf.get_variable("AttnW_%d" % a,
                                [1, 1, attn_size, attention_vec_size])
            hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(tf.get_variable("AttnV_%d" % a,
                                     [attention_vec_size]))

        def attention(query):
            attn_weights = []
            ds = []  # Results of attention reads will be stored here.
            for i in range(num_heads):
                with tf.variable_scope("Attention_%d" % i):
                    y = linear(query, attention_vec_size, True)
                    y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(
                        v[i] * tf.tanh(hidden_features[i] + y), [2, 3])
                    a = tf.nn.softmax(s)
                    attn_weights.append(a)
                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(
                        tf.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                        [1, 2])
                    ds.append(tf.reshape(d, [-1, attn_size]))
            return attn_weights, ds

        batch_attn_size = tf.stack([batch_size, attn_size])
        attns = [tf.zeros(batch_attn_size, dtype=dtype)
                 for _ in range(num_heads)]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])

        # loop through the encoder_outputs
        attention_encoder_outputs = list()
        sequence_attention_weights = list()
        for i in range(timestep_size):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            if i == 0:
                with tf.variable_scope("Initial_Decoder_Attention"):
                    initial_state = linear(encoder_state, output_size, True)
                attn_weights, ds = attention(initial_state)
            else:
                attn_weights, ds = attention(encoder_outputs[i])
            output = tf.concat([ds[0], encoder_outputs[i]], 1)
            # NOTE: here we temporarily assume num_head = 1
            with tf.variable_scope("AttnRnnOutputProjection"):
                logit = linear(output, num_decoder_symbols, True)
            attention_encoder_outputs.append(tf.expand_dims(logit, 1))
            # NOTE: here we temporarily assume num_head = 1
            sequence_attention_weights.append(attn_weights[0])
            # NOTE: here we temporarily assume num_head = 1

        attention_encoder_outputs = tf.concat(attention_encoder_outputs, axis=1)
    return attention_encoder_outputs, sequence_attention_weights
