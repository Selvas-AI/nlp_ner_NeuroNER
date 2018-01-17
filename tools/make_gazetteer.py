# -*- coding: utf-8 -*-
import glob
import json
import pickle

import sys


def loadjsondict(path):
    total_dict = {}
    path_list = glob.glob(path + '/*.txt')
    for p in path_list:
        with open(p, 'r', encoding='utf-8') as f:
            cd = json.load(f)
            for k, v in cd.items():
                if " (" in k:
                    ks = k.split(" (")
                    ks[-1] = ks[-1].strip(')')
                else:
                    ks = [k]
                for kk in ks:
                    kk = kk.replace(' ', '')
                    if kk in total_dict:
                        total_dict[kk].append(v)
                    else:
                        total_dict[kk] = [v]
    return total_dict, max([len(k) for k in total_dict.keys()])

if __name__ == '__main__':
    dict_path = sys.argv[1]
    gazetteer_path = sys.argv[2]
    dic, max_key_len = loadjsondict(dict_path)
    pickle.dump({"dic": dic, "max_key_len": max_key_len}, open(gazetteer_path, "wb"))

