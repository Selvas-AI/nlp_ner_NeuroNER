# -*- coding: utf-8 -*-

import re

import kma.kma as kma

kma.init('kordict')

def pos_analyze(body):
    output = []
    eoj_list = body.split(" ")
    for eoj in eoj_list:
        pos_raw = kma.pos(eoj)
        output.append([[pos_raw[index].rstrip("\x00"), pos_raw[index + 1].rstrip("\x00")] for index in
                   range(0, len(pos_raw), 3)])

    for eoj_index in range(len(eoj_list)):
        eoj = eoj_list[eoj_index]
        pos_list = output[eoj_index]
        last_index = 0
        for pos_index in range(len(pos_list)):
            pos = pos_list[pos_index]
            '''
            if pos_index != 0 and pos_list[pos_index-1][1][0] == 'N'and pos[1][0] != 'N':
                fixed = pos_list[:pos_index-1]
                fixed.append([eoj[sum([len(p[0]) for p in pos_list[:pos_index-1]]):], "EF"])
                output[eoj_index] = fixed
                break
            '''
            try:
                if eoj[last_index:last_index + len(pos[0])] != pos[0]:
                    raise Exception
            except Exception as e:
                fixed = pos_list[:pos_index]
                if len(eoj[last_index:]) != 0:
                    fixed.append([eoj[last_index:], "EF"])
                output[eoj_index] = fixed
                break
            last_index += len(pos[0])
        pos_list = output[eoj_index]
        pos_eoj = "".join([pos[0] for pos in pos_list])
        if len(eoj) > len(pos_eoj):
            output[eoj_index].append([eoj[len(pos_eoj):], "EF"])
    return output
