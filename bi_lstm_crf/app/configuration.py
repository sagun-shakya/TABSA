# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 11:46:27 2022

@author: Sagun Shakya
"""

class Arguments:
    '''
    Use this if you are using IDE for training.
    
    Usage:
        from configuration import Arguments
        args = Arguments(corpus_dir = <path to "sample_corpus">)
        train(args)
    '''
    def __init__(self, corpus_dir, 
                 model_dir = 'model_dir', 
                 num_epoch = 20,
                 lr = 1e-3,
                 weight_decay = 0,
                 batch_size = 1000,
                 device = None, 
                 max_seq_len = 100,
                 val_split = 0.2,
                 test_split = 0.2,
                 recovery = False,
                 save_best_val_model = True,
                 embedding_dim = 100,
                 hidden_dim = 128,
                 num_rnn_layers = 1,
                 rnn_type = 'lstm'
                 ):
        
        self.corpus_dir = corpus_dir
        self.model_dir = model_dir
        self.num_epoch = num_epoch
        self. lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = device
        self.max_seq_len = max_seq_len
        self.val_split = val_split
        self.test_split = test_split
        self.recovery = recovery
        self.save_best_val_model = save_best_val_model
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_rnn_layers = num_rnn_layers
        self.rnn_type = rnn_type
        