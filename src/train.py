import codecs
import os

import sklearn.metrics
import tensorflow as tf

import utils_nlp
from evaluate import remap_labels


# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def train_step(sess, feed_dict, model):
    _, _, loss, accuracy, transition_params_trained = sess.run(
        [model.train_op, model.global_step, model.loss, model.accuracy, model.transition_parameters],
        feed_dict)
    return transition_params_trained


def prediction_step(sess, dataset, dataset_type, model, transition_params_trained, stats_graph_folder, epoch_number,
                    parameters, dataset_filepaths):
    if dataset_type == 'deploy':
        print('Predict labels for the {0} set'.format(dataset_type))
    else:
        print('Evaluate model on the {0} set'.format(dataset_type))
    all_predictions = []
    all_y_true = []
    output_filepath = os.path.join(stats_graph_folder, '{1:03d}_{0}.txt'.format(dataset_type, epoch_number))
    output_file = codecs.open(output_filepath, 'w', 'UTF-8')

    step = 0
    while True:
        if step >= dataset.sample_size[dataset_type]:
            break
        seq_lengths, label_indicess, token_indicess, pos_sequences, space_sequences, token_lengthss, character_indicess, original_conlls = \
        dataset.data_queue[dataset_type].next()

        feed_dict = {
            model.input_token_indices: token_indicess,
            model.input_token_space_indices: space_sequences,
            model.input_token_morpheme_indices: pos_sequences,
            model.input_token_character_indices: character_indicess,
            model.input_token_lengths: token_lengthss,
            model.input_label_indices_flat: label_indicess,
            model.input_seq_lengths: seq_lengths,
            model.training: False,
            model.dropout_keep_prob: 1
        }

        batch_unary_scores, batch_predictions = sess.run([model.unary_scores, model.predictions], feed_dict)
        for idx in range(int(parameters['batch_size'])):
            unary_scores = batch_unary_scores[idx]
            predictions = batch_predictions[idx]
            if parameters['use_crf']:
                predictions, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params_trained)
                predictions = predictions[1:seq_lengths[idx] + 1]
            else:
                predictions = predictions[:seq_lengths[idx]].tolist()

            # assert (len(predictions) == len(dataset.tokens[dataset_type][i]))
            output_string = ''
            prediction_labels = [dataset.index_to_label[prediction] for prediction in predictions]
            gold_labels = [dataset.index_to_label[t] for t in
                           label_indicess[idx][:seq_lengths[idx]]]
            if parameters['tagging_format'] == 'bioes':
                prediction_labels = utils_nlp.bioes_to_bio(prediction_labels)
                gold_labels = utils_nlp.bioes_to_bio(gold_labels)
            tokens = [dataset.index_to_token[t] for t in
                      token_indicess[idx][:seq_lengths[idx]]]
            conll_text = original_conlls[idx].split('\n')
            for line, prediction, token, gold_label in zip(conll_text, prediction_labels, tokens, gold_labels):
                split_line = line.strip().split(' ')
                if parameters['tagging_format'] == 'bioes':
                    split_line.pop()
                gold_label_original = split_line[-1]
                # token_original = split_line[0]
                # assert ((token == 'UNK' or token == token_original) and gold_label == gold_label_original)
                assert (gold_label == gold_label_original)
                split_line.append(prediction)
                output_string += ' '.join(split_line) + '\n'
            output_file.write(output_string + '\n')

            all_predictions.extend(predictions)
            all_y_true.extend(label_indicess[idx][:seq_lengths[idx]])
        step += int(parameters['batch_size'])
    output_file.close()

    if dataset_type != 'deploy':
        if parameters['main_evaluation_mode'] == 'conll':
            conll_evaluation_script = os.path.join('.', 'conlleval')
            conll_output_filepath = '{0}_conll_evaluation.txt'.format(output_filepath)
            shell_command = 'perl {0} < {1} > {2}'.format(conll_evaluation_script, output_filepath,
                                                          conll_output_filepath)
            os.system(shell_command)
            with open(conll_output_filepath, 'r') as f:
                classification_report = f.read()
                print(classification_report)
        else:
            new_y_pred, new_y_true, new_label_indices, new_label_names, _, _ = remap_labels(all_predictions, all_y_true,
                                                                                            dataset, parameters[
                                                                                                'main_evaluation_mode'])
            print(sklearn.metrics.classification_report(new_y_true, new_y_pred, digits=4, labels=new_label_indices,
                                                        target_names=new_label_names))

    return all_predictions, all_y_true, output_filepath


def predict_labels(sess, model, transition_params_trained, parameters, dataset, epoch_number, stats_graph_folder,
                   dataset_filepaths):
    # Predict labels using trained model
    # Predict labels using trained model
    y_pred = {}
    y_true = {}
    output_filepaths = {}
    # for dataset_type in ['train', 'valid', 'test', 'deploy']:
    for dataset_type in ['valid', 'test', 'deploy']:
        if dataset_type not in dataset.data_queue:
            continue
        prediction_output = prediction_step(sess, dataset, dataset_type, model, transition_params_trained,
                                            stats_graph_folder, epoch_number, parameters, dataset_filepaths)
        y_pred[dataset_type], y_true[dataset_type], output_filepaths[dataset_type] = prediction_output
    return y_pred, y_true, output_filepaths
