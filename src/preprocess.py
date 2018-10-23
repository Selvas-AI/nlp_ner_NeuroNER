# -*- coding: utf-8-*-
import re
from collections import namedtuple

import utils
import utils_nlp
from params import PADDING_LABEL_INDEX, PADDING_CHARACTER_INDEX, PADDING_TOKEN_INDEX, LIMIT_SEQUENCE_LENGTH
from morph_analyzer.oktpy.twitter import TwitterMorphManager

ModelInput = namedtuple('ModelInput',
                        'token_indices character_indices label_indices extended_sequence conll token_lengths sequence_length')


def encode(metadata, token_sequence, extended_sequence, label_sequence=None, conll_txt=None, expanded_embedding=None):
    if label_sequence is not None:
        label_indices = []
        for label in label_sequence:
            if label in metadata['label_to_index']:
                label_indices.append(metadata['label_to_index'][label])
            else:
                label_indices.append(PADDING_LABEL_INDEX)
    else:
        label_indices = None

    token_indices = []
    if expanded_embedding is not None:
        for token in token_sequence:
            if token in expanded_embedding:
                token_indices.append(expanded_embedding[token])
            else:
                token_indices.append(expanded_embedding["UNK"])
    else:
        for token in token_sequence:
            if token in metadata['token_to_index']:
                token_indices.append(metadata['token_to_index'][token])
            else:
                token_indices.append(PADDING_TOKEN_INDEX)

    character_indices = []
    for token in token_sequence:
        current_token_character_indices = []
        for character in token:
            if character in metadata['character_to_index']:
                current_token_character_indices.append(metadata['character_to_index'][character])
            else:
                current_token_character_indices.append(PADDING_CHARACTER_INDEX)
        character_indices.append(current_token_character_indices)
    token_lengths = [len(token) for token in token_sequence]
    sequence_length = len(token_indices)
    return ModelInput(token_indices, character_indices, label_indices, extended_sequence, conll_txt,
                      token_lengths, sequence_length)


def pad_and_batch(inputs, batch_size, metadata, is_train=False, expanded_embedding=None, pad_constant_size=False):
    batches = []
    for i in range(0, len(inputs), batch_size):
        batch_data = {}
        begin = i
        end = min(i + batch_size, len(inputs))

        current = inputs[begin:end]

        if pad_constant_size:
            max_token_size = LIMIT_SEQUENCE_LENGTH
        else:
            max_token_size = max([len(element.token_indices) for element in current])

        batch_data['token_indices'] = []
        if expanded_embedding is not None:
            for element in current:
                batch_data['token_indices'].append(
                    utils.pad_list(element.token_indices, max_token_size, expanded_embedding["UNK"]))
        else:
            for element in current:
                '''
                if is_train:
                    for j, token_index in enumerate(element.token_indices):
                        if token_index in metadata['infrequent_token_indices'] and np.random.uniform() < 0.5:
                            element.token_indices[j] = PADDING_TOKEN_INDEX
                '''
                batch_data['token_indices'].append(
                    utils.pad_list(element.token_indices, max_token_size, PADDING_TOKEN_INDEX))

        max_extended_feature_size = len(current[0].extended_sequence[0])
        batch_data['extended_sequence'] = []
        for element in current:
            batch_data['extended_sequence'].append(
                utils.pad_list(element.extended_sequence, max_token_size, [0] * max_extended_feature_size))

        max_char_size = 0
        for element in current:
            for word in element.character_indices:
                max_char_size = max(max_char_size, len(word))

        batch_data['character_indices'] = []
        for element in current:
            word_list = []
            for word in element.character_indices:
                word_list.append(utils.pad_list(word, max_char_size, PADDING_CHARACTER_INDEX))
            batch_data['character_indices'].append(
                utils.pad_list(word_list, max_token_size, [PADDING_CHARACTER_INDEX] * max_char_size))

        batch_data['token_lengths'] = []
        for element in current:
            batch_data['token_lengths'].append(
                utils.pad_list(element.token_lengths, max_token_size, 0))

        batch_data['seq_lengths'] = []
        for element in current:
            batch_data['seq_lengths'].append(element.sequence_length)

        if current[0].label_indices is not None:
            batch_data['label_indices'] = []
            for element in current:
                batch_data['label_indices'].append(utils.pad_list(element.label_indices, max_token_size, 0))

        if current[0].conll is not None:
            batch_data['conll'] = []
            for element in current:
                batch_data['conll'].append(element.conll)

        batch_data['batch_size'] = len(current)

        if len(current) < batch_size:
            for key, value in batch_data.items():
                utils.pad_by_first_element_if_insufficient(value, batch_size)

        batches.append(batch_data)
    return batches


def normalize_token(token):
    token = re.sub('\d+', '0', token)
    #return "".join(TwitterMorphManager().morph_analyzer.morphs(token, stem=True))
    return token


POS_TO_INDICIES = {
    "Noun": 1,
    "Verb": 2,
    "Adjective": 3,
    "Adverb": 4,
    "Determiner": 5,
    "Exclamation": 6,
    "Josa": 7,
    "Eomi": 8,
    "PreEomi": 9,
    "Conjunction": 10,
    "Modifier": 11,
    "VerbPrefix": 12,
    "Suffix": 13,
    "Unknown": 14,
    "Korean": 15,
    "Foreign": 16,
    "Number": 17,
    "KoreanParticle": 18,
    "Alpha": 19,
    "Punctuation": 20,
    "Hashtag": 21,
    "ScreenName": 22,
    "Email": 23,
    "URL": 24,
    "CashTag": 25,
    "Space": 26,
    "Others": 27,
    "ProperNoun;": 28}


def extract_feature(src, tokenizer, gazetteer, max_key_len):
    token_sequence = []
    raw_token_sequence = []
    extended_sequence = []
    if tokenizer == 'pos':
        pos_list = TwitterMorphManager().morph_analyzer.pos(src)
        if gazetteer is not None:
            morphs = []
            for pos in pos_list:
                if pos.pos == 'Space':
                    continue
                morphs.append(pos.text)
            gazetteer_info = utils_nlp.tag_nes(gazetteer, max_key_len, morphs)

        elem_idx = 0
        for idx, pos in enumerate(pos_list):
            if pos.pos == 'Space':
                continue
            token = normalize_token(pos.text)
            raw_token_sequence.append(pos.text)
            token_sequence.append(token)
            pos_tag = POS_TO_INDICIES[pos.pos]
            next_is_space = 0 if idx < len(pos_list) - 1 and pos_list[idx + 1].pos != 'Space' else 1
            if gazetteer is None:
                extended_sequence.append([pos_tag, next_is_space])
            else:
                gz = 1 if len(gazetteer_info[elem_idx]) > 0 else 0
                extended_sequence.append([pos_tag, next_is_space, gz])
            elem_idx += 1

    elif tokenizer == 'character':
        for idx, char in enumerate(src):
            if char == ' ':
                continue
            token = normalize_token(char)
            raw_token_sequence.append(char)
            token_sequence.append(token)
            next_is_space = 0 if idx < len(src) - 1 and src[idx + 1] != ' ' else 1
            extended_sequence.append([next_is_space])
    else:
        raise Exception("")
    return token_sequence, raw_token_sequence, extended_sequence
