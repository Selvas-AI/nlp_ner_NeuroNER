# -*- coding: utf-8-*-
from __future__ import print_function

import argparse
import json
import re
import uuid
import warnings

import os
import six
import sys
from flask import Flask, jsonify
from flask_restful import Api, reqparse, Resource

from neuroner import NeuroNER

warnings.filterwarnings('ignore')
nn = NeuroNER(os.path.dirname(os.path.abspath(__file__)) + "/parameters.ini", predict_mode=True)


class Message(Resource):
    message_pattern = re.compile("(<([A-z\s]+)\s{0,2}:([^<]+|[^>]+)>)")
    BUFF_SIZE = 2048
    NULL = "\x00"

    def post(self):
        print("in 1")
        parser = reqparse.RequestParser()
        parser.add_argument('user_id', type=str, required=True, location='args')
        parser.add_argument('msg', type=str, required=True, location='json')
        parser.add_argument('msg_id', type=str, required=False, default=uuid.uuid1().hex, location='json')
        parser.add_argument('timezone', type=str, required=False, location='json')  # , default="Asia/Seoul")
        parser.add_argument('gps', action='append', required=False, location='json')
        parser.add_argument('additional_info', type=dict, required=False, location='json')
        args = parser.parse_args(strict=True)
        posed, entities = nn.predict(args['msg'])
        raw_response = ""
        raw_response += "<형태소 결과>\n" + str(posed) + "\n\n<개체명 결과>\n"
        if len(entities) == 0:
            raw_response += "none"
        else:
            for idx, entity in enumerate(entities):
                raw_response += "{}. {} - {}\n".format(idx, entity['text'], entity['type'])
        response = self.parse_to_json(raw_response, args['msg_id'])
        return response

    def parse_to_json(self, msg, msg_id):
        response = {}
        response['msg_id'] = msg_id
        response['utterances'] = []

        separated = msg.split('<separator>')
        for separated_msg in separated:
            if separated_msg.strip() == '':
                continue
            recommend_utterance = []
            utterance = {}

            for elem in re.finditer(self.message_pattern, separated_msg):
                elem_text = elem.group(0)
                key = elem.group(2).strip(' ')
                value = elem.group(3)

                if key == 'recommend':
                    recommend_utterance.append(value.strip(" "))
                else:
                    try:
                        current_data = json.loads(value)
                        if key in utterance and isinstance(utterance[key], dict):
                            utterance[key].update(current_data)
                        else:
                            utterance[key] = current_data
                    except ValueError as e:
                        if key in utterance and isinstance(utterance[key], six.string_types):
                            utterance[key] += value
                        else:
                            utterance[key] = value
                        pass
                separated_msg = separated_msg.replace(elem_text, "")
            separated_msg = separated_msg.strip("\n")
            separated_msg = separated_msg.strip(" ")
            utterance['text'] = separated_msg
            utterance['recommend_utterance'] = recommend_utterance
            response['utterances'].append(utterance)
        return jsonify(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    conf = parser.parse_args()
    # init ##
    app = Flask(__name__)
    api = Api(app)

    api.add_resource(Message, '/message')

    app.run(host='0.0.0.0', port=conf.port, debug=False, threaded=False)
