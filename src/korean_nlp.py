# -*- coding: utf-8 -*-

from twitter.pyokt.twitter import Twitter

twitter = Twitter()


def pos_analyze(sentence):
    output = []
    eoj = []
    pos_list = twitter.pos(sentence)
    for pos in pos_list:
        if pos.pos == 'Space':
            output.append(eoj)
            eoj = []
        else:
            eoj.append([pos.text, pos.pos])
    output.append(eoj)
    return output
