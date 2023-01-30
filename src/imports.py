from utils.contants import CONSTANTS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup

import numpy as np
import pandas as pd
import tqdm
import itertools
from abc import ABC, abstractmethod

import logging
import time
logging.basicConfig(
    filename=f'./Projects/RESQU_BERT/{CONSTANTS.LOG_PATH}',
    encoding='utf-8',
    level=logging.DEBUG
)
logging.info(f"Start logging at: {time.ctime()}")
