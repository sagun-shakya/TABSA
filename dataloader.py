# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 09:33:47 2021

@author: Sagun Shakya
"""
'''
TO-DO:
    - Add Embedding matrix. see https://github.com/oya163/nepali-sentiment-analysis/blob/master/utility/preprocess/word_vectors.py
'''
class DataLoaderConstant:
    def __init__(self, vocab_size, tagset_size, weights):
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.weights = weights
