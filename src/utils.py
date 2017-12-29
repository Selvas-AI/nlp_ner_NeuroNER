import collections
import datetime
import glob
import os
import shutil
import time

import numpy as np


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    else:
        return False


def convert_configparser_to_dictionary(config):
    my_config_parser_dict = {s: dict(config.items(s)) for s in config.sections()}
    return my_config_parser_dict


def load_pretrained_token_embeddings(embedding_filepath):
    file_input = open(embedding_filepath, 'r', encoding='UTF-8')
    count = -1
    token_to_vector = {}
    for cur_line in file_input:
        count += 1
        # if count > 1000:break
        cur_line = cur_line.strip()
        cur_line = cur_line.split(' ')
        if len(cur_line) == 0: continue
        token = cur_line[0]
        vector = np.array([float(x) for x in cur_line[1:]])
        token_to_vector[token] = vector
    file_input.close()
    return token_to_vector


def pad_list(old_list, padding_size, padding_value):
    if padding_size == len(old_list):
        return old_list
    assert padding_size >= len(old_list)
    if type(old_list) is np.ndarray:
        return np.append(old_list, [padding_value] * (padding_size - len(old_list)), axis=0)
    else:
        return old_list + [padding_value] * (padding_size - len(old_list))


def reverse_dictionary(dictionary):
    if type(dictionary) is collections.OrderedDict:
        return collections.OrderedDict([(v, k) for k, v in dictionary.items()])
    else:
        return {v: k for k, v in dictionary.items()}


def create_folder_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_current_milliseconds():
    return (int(round(time.time() * 1000)))


def get_current_time_in_seconds():
    return (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))


def get_current_time_in_miliseconds():
    return (get_current_time_in_seconds() + '-' + str(datetime.datetime.now().microsecond))


def get_basename_without_extension(filepath):
    return os.path.basename(os.path.splitext(filepath)[0])


def pad_by_first_element_if_insufficient(element, size):
    if type(element) is not list or len(element) == size:
        return
    remain = size - len(element)
    for _ in range(remain):
        element.append(element[0])


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def get_valid_dataset_filepaths(data_path, dataset_types=['train', 'valid', 'test']):
    dataset_filepaths = collections.OrderedDict()

    for dataset_type in dataset_types:
        target_path = os.path.join(data_path, dataset_type)
        if os.path.exists(target_path) and len(list(glob.glob(target_path + "/*.txt"))) > 0:
            dataset_filepaths[dataset_type] = target_path
    return dataset_filepaths