# -*- coding: utf-8-*-
import argparse
import glob
import os
import time
import warnings

from oktpy.twitter import TwitterMorphManager
from tqdm import tqdm

import utils
from metadata import Metadata
from kor_neuroner import KorNeuroNER
from params import Configuration
from multiprocessing.pool import Pool

warnings.filterwarnings('ignore')
TwitterMorphManager().morph_analyzer.pos("안녕하세요")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameter', type=str, default='./parameters.ini')
    parser.add_argument('--corpus_path', type=str)
    parser.add_argument('--target_path', type=str)
    cmd_arg = parser.parse_args()

    start_time = time.time()
    print('Init... ', end='', flush=True)
    parameters = Configuration(cmd_arg.parameter)

    parameters['mode'] = 'test'
    parameters['batch_size'] = 128
    parameters['enable_tensorbord'] = False
    metadata = Metadata(parameters['pretrained_model_folder'], None, None, None)
    neuroner = KorNeuroNER(parameters, metadata)
    print('done ({0:.2f} seconds)'.format(time.time() - start_time))

    corpus_list = glob.iglob(cmd_arg.corpus_path + "/*.txt", recursive=True)
    experiment_timestamp = utils.get_current_time_in_miliseconds()
    output_index = 0
    output_path = cmd_arg.target_path + "/" + experiment_timestamp + "_%d.txt"

    output_file = open(output_path % output_index, "w", encoding='utf-8')
    write_count = 0
    pool = Pool(5)
    for file_path in tqdm(list(corpus_list)):
        remain_txt = ""
        with open(file_path, "r", encoding="utf-8") as input_file:
            line_list = input_file.readlines()
            line_list = [line.strip("\n") for line in line_list]
            line_list = [line for line in line_list if len(line.split(" ")) < 50]

            raw_token_sequence_list, extended_sequence_list, label_sequence_list, score_list = neuroner.predict_list(
                line_list, pool)
            for idx in range(len(line_list)):
                raw_token_sequence = raw_token_sequence_list[idx]
                extended_sequence = extended_sequence_list[idx]
                label_sequence = label_sequence_list[idx]
                score = score_list[idx]

                score /= len(raw_token_sequence)
                if (score < 17) or (len(set(label_sequence)) == 1 and label_sequence[0] == "O"):
                    if len(remain_txt) != 0:
                        remain_txt += "\n"
                    remain_txt += line_list[idx]
                    continue
                index = 0
                for tok, extended, label in zip(raw_token_sequence, extended_sequence, label_sequence):
                    elem = "{} . {} {} {} {}\n".format(tok, index, index + len(tok),
                                                       " ".join([str(ex) for ex in extended]), label)
                    index += len(tok) + extended[-1]
                    output_file.write(elem)
                output_file.write("\n")
                output_file.flush()
                write_count += 1
                if write_count > 10000:
                    write_count = 0
                    output_file.close()
                    output_index += 1
                    output_file = open(output_path % output_index, "w", encoding='utf-8')

        if len(remain_txt) != 0:
            with open(file_path, "w", encoding="utf-8") as input_file:
                input_file.write(remain_txt)
        else:
            os.remove(file_path)

    neuroner.close()
    print('complete ({0:.2f} seconds)'.format(time.time() - start_time))


if __name__ == "__main__":
    main()
