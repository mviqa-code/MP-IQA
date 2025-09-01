import functools
import logging
import sys
import os
import random
from datetime import datetime

import torch
import numpy as np
from termcolor import colored


@functools.lru_cache()
def create_logger(logger, output_dir):
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_formatter = colored('[%(asctime)s %(name)s]', 'green') + \
                      colored('(%(filename)s %(lineno)d)', 'yellow') + \
                      ': %(levelname)s %(message)s'

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_formatter, datefmt='%Y-%m-%d %H:%M:%S')
    )
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(output_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_log.txt'),
                                       mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=formatter, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)