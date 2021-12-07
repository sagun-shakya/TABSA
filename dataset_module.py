# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 08:14:58 2021

@author: Sagun Shakya
"""
import pandas as pd
from torch.utils.data import Dataset
import torch


# LOCAL MODULES.
import utils

class TABSADataset(Dataset):
    """
    This is a custom dataset class. It gets the X and Y data to be fed into the Dataloader.
    """
    def __init__(self, X, Y, pad_id):
        self.X = X
        self.Y = Y
        self.pad_id = pad_id
        if len(self.X) != len(self.Y):
            raise Exception("The length of X does not match the length of Y.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y.
        _y = self.Y[index]
        
        _x = self.X[index]
        _l = len(tuple(filter(lambda x: x != self.pad_id, self.X[index])))  # Calculates the length of the original unpadded sentence.

        return (_x,_l), _y


class TrainTest:
    def __init__(self, data):
                
        # Unique self.words.
        self.words = data['token'].unique().tolist() + ['<UNK>', '<PAD>', 'ENDTAG']
        
        # Unique self.tags.
        self.tags = data['tag'].unique().tolist()
        
        # Grouped data.
        grouped = data.groupby('sentence_no')
        
        # Aggregation function to make a list of tokens in each row.
        agg_function_word_only = lambda df: [w for w in df['token'].values.tolist()]
        
        # List of word_lists.
        self.sentences = grouped.apply(agg_function_word_only).tolist()
        
        # Aggregation function to make a list of self.self.tags in each row.
        agg_function_tag_only = lambda df: [t for t in df['tag'].values.tolist()]
                
        # List of self.tags_tokens.
        self.tag_sequence = grouped.apply(agg_function_tag_only).tolist()
        
        # DataFrame for sentences and self.tags.
        self.df_sent_tags = pd.DataFrame({'sentences' : self.sentences, 'tag_sequence' : self.tag_sequence})
        
        # Remove rows for len less than 3.
        id_del = [ii for ii, item in enumerate(self.sentences) if len(item) < 3]
        self.df_sent_tags.drop(self.df_sent_tags.index[id_del], inplace=True)
        
        print("Number of sentences having less than 3 self.words (that is removed): ", len(self.sentences) - len(self.df_sent_tags))
        
        # Mappings from word/tag to its IDs.
        self.word2idx = {word : ii for ii, word in enumerate(self.words, 1)}
        self.tag2idx = {tag : ii for ii, tag in enumerate(self.tags, 0)}
        self.tag_pad_id = len(self.tags)
        self.tag2idx['<pad>'] = self.tag_pad_id
        
        # Inverse mapping.
        self.idx2word = {ii : word for ii, word in enumerate(self.words, 1)}
        self.idx2tag = {ii : tag for ii, tag in self.tag2idx.items()}
        self.idx2tag[self.tag_pad_id] = '<pad>'
        
        # Pad Index for word_sequence and tag_sequence.
        self.word_pad_id = self.word2idx['<PAD>']
        
        # Unpadded sequences.
        # List of token IDs for wach sentence and the self.tagset. 
        self.df_sent_tags['word_id'] = self.df_sent_tags['sentences'].apply(lambda x: torch.tensor(utils.word_list2id_list(x, 
                                                                                                                           self.word2idx, 
                                                                                                                           self.tag2idx, 
                                                                                                                           mapper = 'word'))) 
        self.df_sent_tags['seq_id'] = self.df_sent_tags['tag_sequence'].apply(lambda x: torch.tensor(utils.word_list2id_list(x, 
                                                                                                                             self.word2idx, 
                                                                                                                             self.tag2idx, 
                                                                                                                             mapper = 'tag'))) 
        
        
        # x.
        self.df_sent_tags['padded_word_id'] = self.df_sent_tags['word_id'].apply(lambda x: utils.padder_func(x, self.word_pad_id))
        # y.
        self.df_sent_tags['padded_seq_id'] = self.df_sent_tags['seq_id'].apply(lambda x: utils.padder_func(x, self.tag_pad_id))
        
