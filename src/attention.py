# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops

try:
    from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear as linear
except Exception:
    from tensorflow.python.ops.rnn_cell_impl import _linear as linear


def attention_decoder(encoder_outputs_raw,
                      encoder_state,
                      decoder_output_size,
                      timestep_size,
                      num_heads=1,
                      dtype=tf.float32,
                      scope=None):
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")

    with tf.variable_scope(scope or "attention_RNN"):
        encoder_outputs = array_ops.transpose(encoder_outputs_raw, [1, 0, 2])
        encoder_output_size = encoder_outputs.get_shape()[2].value
        top_states = [tf.reshape(encoder_outputs[ti], [-1, 1, encoder_output_size])
                      for ti in range(timestep_size)]
        attention_states = tf.concat(top_states, 1)
        if not attention_states.get_shape()[1:2].is_fully_defined():
            raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                             % attention_states.get_shape())

        batch_size = tf.shape(top_states[0])[0]  # Needed for reshaping.
        if attention_states.get_shape()[1].value != timestep_size:
            raise ValueError("timestep size must be constant")
        attn_size = encoder_output_size

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = tf.reshape(
            attention_states, [-1, timestep_size, 1, attn_size])
        hidden_features = []
        v = []
        for a in range(num_heads):
            k = tf.get_variable("AttnW_%d" % a,
                                [1, 1, attn_size, decoder_output_size])
            hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(tf.get_variable("AttnV_%d" % a,
                                     [decoder_output_size]))

        def attention(query):
            attn_weights = []
            ds = []  # Results of attention reads will be stored here.
            for i in range(num_heads):
                with tf.variable_scope("Attention_%d" % i):
                    y = linear(query, decoder_output_size, True)
                    y = tf.reshape(y, [-1, 1, 1, decoder_output_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(
                        v[i] * tf.tanh(hidden_features[i] + y), [2, 3])
                    a = tf.nn.softmax(s)
                    attn_weights.append(a)
                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(
                        tf.reshape(a, [-1, timestep_size, 1, 1]) * hidden,
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
            if i == 0:
                with tf.variable_scope("Initial_Decoder_Attention"):
                    initial_state = linear(encoder_state, decoder_output_size, True)
                attn_weights, ds = attention(initial_state)
            else:
                tf.get_variable_scope().reuse_variables()
                attn_weights, ds = attention(logit)
            output = tf.concat([ds[0], encoder_outputs[i]], 1)
            # NOTE: here we temporarily assume num_head = 1
            with tf.variable_scope("AttnRnnOutputProjection"):
                logit = linear(output, decoder_output_size, True)
            attention_encoder_outputs.append(tf.expand_dims(logit, 1))
            # NOTE: here we temporarily assume num_head = 1
            sequence_attention_weights.append(attn_weights[0])
            # NOTE: here we temporarily assume num_head = 1

        attention_encoder_outputs = tf.concat(attention_encoder_outputs, axis=1)
    return attention_encoder_outputs, sequence_attention_weights
