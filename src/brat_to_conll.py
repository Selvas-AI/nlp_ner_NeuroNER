# -*- coding: utf-8 -*-
import codecs
import glob
import json
import multiprocessing
import os
import string
from functools import partial
from multiprocessing.pool import ThreadPool

import re
import spacy
import psutil
import utils_nlp
from pycorenlp import StanfordCoreNLP

from korean_nlp import pos_analyze
from tqdm import tqdm
from colorama import init


MAX_SENTNECE_LEN = 500
MAX_WORD_LEN = 20

def get_start_and_end_offset_of_token_from_spacy(token):
    start = token.idx
    end = start + len(token)
    return start, end


def get_sentences_and_tokens_from_spacy(text, spacy_nlp):
    document = spacy_nlp(text)
    # sentences
    sentences = []
    for span in document.sents:
        sentence = [document[i] for i in range(span.start, span.end)]
        sentence_tokens = []
        for token in sentence:
            token_dict = {}
            token_dict['start'], token_dict['end'] = get_start_and_end_offset_of_token_from_spacy(token)
            token_dict['text'] = text[token_dict['start']:token_dict['end']]
            if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token_dict['text'].split(' ')) != 1:
                print(
                    "WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(
                        token_dict['text'],
                        token_dict['text'].replace(' ', '-')))
                token_dict['text'] = token_dict['text'].replace(' ', '-')
            sentence_tokens.append(token_dict)
        sentences.append(sentence_tokens)
    return sentences


def get_stanford_annotations(text, core_nlp, port=9000, annotators='tokenize,ssplit,pos,lemma'):
    output = core_nlp.annotate(text, properties={
        "timeout": "10000",
        "ssplit.newlineIsSentenceBreak": "two",
        'annotators': annotators,
        'outputFormat': 'json'
    })
    if type(output) is str:
        output = json.loads(output, strict=False)
    return output


def get_sentences_and_tokens_from_stanford(text, core_nlp):
    stanford_output = get_stanford_annotations(text, core_nlp)
    sentences = []
    for sentence in stanford_output['sentences']:
        tokens = []
        for token in sentence['tokens']:
            token['start'] = int(token['characterOffsetBegin'])
            token['end'] = int(token['characterOffsetEnd'])
            token['text'] = text[token['start']:token['end']]
            if token['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token['text'].split(' ')) != 1:
                print(
                    "WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(
                        token['text'],
                        token['text'].replace(' ', '-')))
                token['text'] = token['text'].replace(' ', '-')
            tokens.append(token)
        sentences.append(tokens)
    return sentences

def parse_line(text_filepath, text, line):
    anno = line.split()
    id_anno = anno[0]
    # parse entity
    entity = {}
    entity['id'] = id_anno
    entity['type'] = anno[1]
    entity['start'] = int(anno[2])
    entity['end'] = int(anno[3])
    entity['text'] = ' '.join(anno[4:])
    # Check compatibility between brat text and anootation
    if utils_nlp.replace_unicode_whitespaces_with_ascii_whitespace(text[entity['start']:entity['end']]) != \
            utils_nlp.replace_unicode_whitespaces_with_ascii_whitespace(entity['text']):
        print("Warning: brat text and annotation do not match.")
        print("file path = %s" % text_filepath)
        print("\ttext: {0}".format(text[entity['start']:entity['end']]))
        print("\tanno: {0}".format(entity['text']))
        raise Exception
    # add to entitys data
    return entity


def get_entities_from_brat(text_filepath, annotation_filepath, verbose=False):
    # load text
    with codecs.open(text_filepath, 'r', 'UTF-8') as f:
        text = f.read()
    if verbose: print("text: {0}".format(text))

    # parse annotation file
    with codecs.open(annotation_filepath, 'r', 'UTF-8') as f:
        entities = [parse_line(text_filepath, text, line) for line in f.read().splitlines() if line[0] == 'T']
    return text, entities


def check_brat_annotation_and_text_compatibility(brat_folder):
    '''
    Check if brat annotation and text files are compatible.
    '''
    dataset_type = os.path.basename(brat_folder)
    print("Checking the validity of BRAT-formatted {0} set... ".format(dataset_type), end='')
    text_filepaths = sorted(glob.glob(os.path.join(brat_folder, '*.txt')))
    for text_filepath in text_filepaths:
        base_filename = os.path.splitext(os.path.basename(text_filepath))[0]
        annotation_filepath = os.path.join(os.path.dirname(text_filepath), base_filename + '.ann')
        # check if annotation file exists
        if not os.path.exists(annotation_filepath):
            raise IOError("Annotation file does not exist: {0}".format(annotation_filepath))
        text, entities = get_entities_from_brat(text_filepath, annotation_filepath)
    print("Done.")


def limit_cpu():
    "is called at every process start"
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    p.nice(psutil.HIGH_PRIORITY_CLASS)


POS_TO_INDICIES = {
"Noun" : 1,
"Verb" : 2,
"Adjective" : 3,
"Adverb" : 4,
"Determiner" : 5,
"Exclamation" : 6,
"Josa" : 7,
"Eomi" : 8,
"PreEomi" : 9,
"Conjunction" : 10,
"NounPrefix" : 11,
"VerbPrefix" : 12,
"Suffix" : 13,
"Unknown" : 14,
"Korean" : 15,
"Foreign" : 16,
"Number" : 17,
"KoreanParticle" : 18,
"Alpha" : 19,
"Punctuation" : 20,
"Hashtag" : 21,
"ScreenName" : 22,
"Email" : 23,
"URL" : 24,
"CashTag" : 25,
"Space" : 26,
"Others" : 27,
"ProperNoun;" : 28}


hangul = re.compile('[^ㄱ-ㅎ가-힣a-zA-Z' + string.punctuation + ']')


def get_sentences_and_tokens_from_korean(text):
    sentences = []
    last_index = 0
    line_size = 0

    for line in text.splitlines():
        '''
        line = " · ".join(line.split('·'))
        line = " ( ".join(line.split('('))
        line = " ) ".join(line.split(')'))
        '''
        #line = line.replace('·', ' ')
        #line = line.replace('(', ' ')
        #line = line.replace(')', ' ')

        break_for_loop = False
        if bool(hangul.match(line)) or len(line) > MAX_SENTNECE_LEN:
            break_for_loop = True
        pos_info = pos_analyze(line)
        sentence_tokens = []

        for pos in pos_info:
            for token_index, token in enumerate(pos):
                token_dict = {}
                token_dict['start'] = last_index
                token_dict['end'] = last_index + len(token[0])
                token_dict['text'] = token[0]
                if len(token[0]) > MAX_WORD_LEN:
                    break_for_loop = True
                token_dict['pos'] = POS_TO_INDICIES[token[1]]

                token_dict['space'] = 1 if token_index == len(pos) - 1 else 0
                if text[token_dict['start']:token_dict['end']] != token_dict['text']:
                    print("")
                    print(line)
                    print(text[token_dict['start']:token_dict['end']])
                    print(token_dict['text'])
                    raise Exception

                last_index = token_dict['end']
                if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                    continue
                # Make sure that the token text does not contain any space
                if len(token_dict['text'].split(' ')) != 1:
                    print(
                        "WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(
                            token_dict['text'],
                            token_dict['text'].replace(' ', '-')))
                    token_dict['text'] = token_dict['text'].replace(' ', '-')
                    raise Exception
                sentence_tokens.append(token_dict)
            last_index += 1
        # last_index += 1
        # 문제를 확인하기 힘들게 될수 있으니, 아래 코드를 제거하고 시간날때 정리해야함
        line_size += len(line) + 1
        last_index = line_size
        if not break_for_loop:
            sentences.append(sentence_tokens)
    return sentences

def brat_to_conll(input_folder, output_filepath, tokenizer, language):
    '''
    Assumes '.txt' and '.ann' files are in the input_folder.
    Checks for the compatibility between .txt and .ann at the same time.
    '''
    init(autoreset=True)
    if tokenizer == 'spacy':
        spacy_nlp = spacy.load(language)
    elif tokenizer == 'stanford':
        core_nlp = StanfordCoreNLP('http://localhost:{0}'.format(9000))
    elif tokenizer == 'korean':
        korean_nlp = None
    else:
        raise ValueError("tokenizer should be either 'spacy' or 'stanford'.")
    verbose = False
    dataset_type = os.path.basename(input_folder)
    #print("Formatting {0} set from BRAT to CONLL... ".format(dataset_type), end='')
    print("Formatting {0} set from BRAT to CONLL... ".format(dataset_type))
    text_filepaths = sorted(glob.glob(os.path.join(input_folder, '*.txt')))
    #text_filepaths=list(glob.iglob('%s/**/*.txt' % input_folder, recursive=True))
    output_file = codecs.open(output_filepath, 'w', 'utf-8')
    for text_filepath in tqdm(text_filepaths, desc='files'):
        base_filename = os.path.splitext(os.path.basename(text_filepath))[0]
        annotation_filepath = os.path.join(os.path.dirname(text_filepath), base_filename + '.ann')
        # create annotation file if it does not exist
        if not os.path.exists(annotation_filepath):
            codecs.open(annotation_filepath, 'w', 'UTF-8').close()

        text, entities = get_entities_from_brat(text_filepath, annotation_filepath)
        entities = sorted(entities, key=lambda entity: entity["start"])

        if tokenizer == 'spacy':
            sentences = get_sentences_and_tokens_from_spacy(text, spacy_nlp)
        elif tokenizer == 'stanford':
            sentences = get_sentences_and_tokens_from_stanford(text, core_nlp)
        elif tokenizer == 'korean':
            sentences = get_sentences_and_tokens_from_korean(text)

        output_text = ""
        hit_map = [None for _ in range(sentences[-1][-1]['end'] + 1)]

        for entity in entities:
            for index in range(entity['start'], entity['end']):
                # Because the ANN doesn't support tag with '-' in it
                #hit_map[index] = entity['type'].replace('-', '_')
                hit_map[index] = entity

        pool = ThreadPool(64, limit_cpu)
        partial_parse_token = partial(parse_token, base_filename=base_filename, entities=entities, hit_map=hit_map, verbose=verbose)
        result = pool.imap(partial_parse_token, sentences)

        for _ in tqdm(range(len(sentences)), desc='sentences'):
            output_file.write(result.next())
            if verbose: print('\n')
            #output_text += '\n'
            output_file.write("\n")
        #output_file.write(output_text)

    output_file.close()
    print('Done.')
    if tokenizer == 'spacy':
        del spacy_nlp
    elif tokenizer == 'stanford':
        del core_nlp


def parse_token(sentence, base_filename, entities, hit_map, verbose):
    output_text = ""
    inside = False
    previous_token_label = 'O'
    for token in sentence:
        if hit_map[token['start']] is not None:
            entity = hit_map[token['start']]
            token['label'] = entity['type'].replace('-', '_')
        elif hit_map[token['end'] - 1] is not None:
            entity = hit_map[token['end'] - 1]
            token['label'] = entity['type'].replace('-', '_')
        else:
            token['label'] = 'O'
            entity = {'end': 0}

        if len(entities) == 0:
            entity = {'end': 0}
        if token['label'] == 'O':
            gold_label = 'O'
            inside = False
        elif inside and token['label'] == previous_token_label:
            gold_label = 'I-{0}'.format(token['label'])
        else:
            inside = True
            gold_label = 'B-{0}'.format(token['label'])
        if token['end'] == entity['end']:
            inside = False
        previous_token_label = token['label']
        if verbose: print(
            '{0} {1} {2} {3} {4} {5} {6}\n'.format(token['text'], base_filename, token['start'], token['end'], token['pos'], token['space'],
                                           gold_label))
        output_text += '{0} {1} {2} {3} {4} {5} {6}\n'.format(token['text'], base_filename, token['start'], token['end'], token['pos'], token['space'],
                                                      gold_label)
    return output_text