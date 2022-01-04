# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 09:30:45 2022

@author: Home
"""
import json
import pandas as pd
from bi_lstm_crf.app.preprocessing.utils import save_json_file, load_json_file
from bi_lstm_crf.app.preprocessing.utils import wordlist_to_idx, padder_func
from preprocess import Preprocessor

corpus_dir_path = r'D:/ML_projects/TABSA/bi-lstm-crf/bi_lstm_crf/app/sample_corpus'
config_dir_path = r'D:/ML_projects/TABSA/bi-lstm-crf/model_dir' 
file_path = r'D:/ML_projects/TABSA/bi-lstm-crf/bi_lstm_crf/app/sample_corpus/dataset1.pkl'
#file_path = r'https://raw.githubusercontent.com/jidasheng/bi-lstm-crf/master/bi_lstm_crf/app/sample_corpus/dataset.txt'


# =============================================================================
# xs, ys = [], []
# with open(file_path, encoding="utf8") as f:
#     for idx, line in enumerate(f):
#         fields = line.split("\t")
#         if len(fields) != 2:
#             raise ValueError("format error in line {}, tabs count: {}".format(idx + 1, len(fields) - 1))
# 
#         sentence, tags = fields
#           
#         
#         try:
#             #if sentence[0] == "[":
#                 #sentence = json.loads(sentence)
#             tags = json.loads(tags)
#             #xs.append(sent_to_vector(sentence, max_seq_len=max_seq_len))
#             #ys.append(tags_to_vector(tags, max_seq_len=max_seq_len))
#             if len(sentence) != len(tags):
#                 raise ValueError('"sentence length({})" != "tags length({})" in line {}"'.format(len(sentence), len(tags), idx + 1))
#         
#         except Exception as e:
#             raise ValueError("exception raised when parsing line {}\n\t{}\n\t{}".format(idx + 1, line, e))
# =============================================================================

def element_get(file_path, offset = 0):
    elements = load_json_file(file_path)
    elements_dict = {w: idx + offset for idx, w in enumerate(elements)}
    return elements, elements_dict

vocab, vocab_dict = element_get(r'D:/ML_projects/TABSA/bi-lstm-crf/bi_lstm_crf/app/sample_corpus/vocab.json', offset = 1)
tags, tags_dict = element_get(r'D:/ML_projects/TABSA/bi-lstm-crf/bi_lstm_crf/app/sample_corpus/tags.json')

PAD_IDX = 0
OOV_IDX = len(vocab)
vocab.insert(0, PAD_IDX)
vocab_dict[PAD_IDX] = 0

vocab.append(OOV_IDX)
vocab_dict[OOV_IDX] = len(vocab) - 1

f = pd.read_pickle(file_path)
xs = f['sentences'].apply(lambda x: padder_func(wordlist_to_idx(x, vocab_dict, OOV_IDX), value = PAD_IDX, max_len = 60))
ys = f['tag_sequence'].apply(lambda x: padder_func(wordlist_to_idx(x, tags_dict, OOV_IDX), value = PAD_IDX, max_len = 60))

xs = xs.values
ys = ys.values

