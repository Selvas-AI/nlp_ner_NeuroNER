import os

file_prefix = r'E:\tech\space\valid\data'
output = r'E:\tech\out'
file_seperate_size = 100000

if __name__ == "__main__":
    basename = os.path.basename(file_prefix)
    srcfile = open(file_prefix + ".source.txt", 'r', encoding='UTF-8')
    tgtfile = open(file_prefix + ".target.txt", 'r', encoding='UTF-8')
    outfilepath = output + "/%04d.txt"
    outfile_index = 0
    line = srcfile.readline()
    tags = tgtfile.readline()

    count = 0
    while line and tags:
        line = line.strip("\n")
        tags = tags.strip("\n")
        if len(line) == 0 or len(tags) == 0 or len(line) != len(tags):
            line = srcfile.readline()
            tags = tgtfile.readline()
            continue

        conll_txt = ""
        for idx, (char, tag) in enumerate(zip(line, tags)):
            if tag == 'D':
                continue
            if tag == 'P':
                label = 'O'
            else:
                label = 'B-SP'
            if len(line) == idx + 1 or tags[idx + 1] == 'D':
                space = 1
            else:
                space = 0
            conll_txt += "{} . {} {} {} {}\n".format(char, idx, idx + 1, space, label)
        conll_txt += "\n"
        if count % file_seperate_size == 0:
            outfile_index += 1
            outfile = open(outfilepath % outfile_index, 'w', encoding='UTF-8')
        outfile.write(conll_txt)
        count += 1

        line = srcfile.readline()
        tags = tgtfile.readline()
