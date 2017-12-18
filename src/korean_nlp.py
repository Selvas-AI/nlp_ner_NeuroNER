# -*- coding: utf-8 -*-

from konlpy.tag import Twitter

twitter = Twitter()


def pos_analyze(sentence):
    output = []
    eoj = []
    pos_list = twitter.pos(sentence)
    begin = 0
    for idx, pos in enumerate(pos_list):
        while True:
            if not sentence[begin:].startswith(pos[0]):
                begin += 1
            break
        end = begin + len(pos[0])
        eoj.append([pos[0], pos[1]])
        if end == len(sentence) or sentence[end] == ' ':
            output.append(eoj)
            eoj = []
        begin = end
    return output