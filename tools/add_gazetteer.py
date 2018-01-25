import glob

import pickle

import utils_nlp

DATA_PATH = r"D:\tech\entity\NeuroNER\data\exobrain3"
pk = pickle.load(open(r"D:\tech\entity\NeuroNER\data\exobrain4\gazeteer", "rb"))
gazetteer = pk['dic']
max_key_len = pk['max_key_len']

file_list = list(glob.iglob(DATA_PATH + '/**/*.txt', recursive=True))
for file_path in file_list:
    outfile = open(file_path + ".mod.txt", 'w', encoding='UTF-8')
    with open(file_path, 'r', encoding='UTF-8') as f:
        file_content = f.read()
        sentences = file_content.split("\n\n")
        for sentence in sentences:
            token_sequence = []
            label_sequence = []
            extended_sequence = []
            conll_txt = ''

            sentence = sentence.strip(" \n")
            if len(sentence) == 0:
                continue
            lines = sentence.split("\n")

            morphs = []
            line_list = []
            for line_raw in lines:
                if '-DOCSTART-' in line_raw:
                    continue
                line = line_raw.strip().split(' ')

                morphs.append(line[0])
                line_list.append(line)

            gazetteer_info = utils_nlp.tag_nes(gazetteer, max_key_len, morphs)


            output_text = ""
            for line, gz in zip(line_list, gazetteer_info):
                line.insert(-1, '1' if len(gz) > 0 else '0')
                output_text += " ".join(line) + "\n"
            output_text += "\n"
            outfile.write(output_text)

