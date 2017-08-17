'''
To run:
CUDA_VISIBLE_DEVICES="" python3.5 main.py &
CUDA_VISIBLE_DEVICES=1 python3.5 main.py &
CUDA_VISIBLE_DEVICES=2 python3.5 main.py &
CUDA_VISIBLE_DEVICES=3 python3.5 main.py &
'''
from __future__ import print_function

import sys
import warnings

from neuroner import NeuroNER

warnings.filterwarnings('ignore')


def main(argv):
    ''' NeuroNER main method

    Args:
        parameters_filepath the path to the parameters file
        output_folder the path to the output folder
    '''
    if len(argv) == 1:
        parameters_filepath = './parameters.ini'
    else:
        parameters_filepath = sys.argv[1]

    nn = NeuroNER(parameters_filepath)
    nn.fit()
    nn.close()


if __name__ == "__main__":
    main(sys.argv)
