# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 09:33:47 2021

@author: Sagun Shakya
"""

class DataLoader:
    def __init__(self, vocab_size, tagset_size, weights):
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.weights = weights
