# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 09:12:53 2021

@author: Sagun Shakya
"""

class Configuration:
    '''Configuration for parameters for the LSTM model.'''
    
    def __init__(self, bidirection, batch_size, num_layers, hidden_dim, embedding_dim, dropout_prob, pretrained, train_type):
        self.bidirection = bidirection
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob
        self.pretrained = pretrained
        self.train_type = train_type
        
        
