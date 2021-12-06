# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:05:52 2021

@author: Sagun Shakya
"""

#Importing necessary libraries.
import os
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.model_selection import train_test_split

# LOCAL MODULES.
from load_data import get_file_in_df
from dataset_module import TABSADataset
from config import Configuration
from dataloader import DataLoaderConstant
from BiLSTM_model import BiLSTM
from dynamic_model import DynamicModel
from EarlyStopping_module import EarlyStopping
import utils

# COnstants.
BATCH_SIZE = 32

# Relative path.
filename = os.path.join(os.path.dirname(__file__), r'data', r'data_aspect_target_extraction.txt')

# Parse the txt file in a Pandas DataFrame.
data = get_file_in_df(filename)

# Removing the last 4 rows.
data = data.iloc[:-4]

# to_remove.
to_remove = {'...', '!', ',', '||', '/', '.', '(', ')', '..', '!', '@', '#', '%', '।'}

#data['token'] = data['token'].map({symbol: np.nan for symbol in to_remove})

# Unique Words.
words = data['token'].unique().tolist() + ['<UNK>', '<PAD>', 'ENDTAG']

# Unique Tags.
tags = data['tag'].unique().tolist()

# Grouped data.
grouped = data.groupby('sentence_no')

# Aggregation function to make a list of tokens in each row.
agg_function_word_only = lambda df: [w for w in df['token'].values.tolist()]

# List of word_lists.
sentences = grouped.apply(agg_function_word_only).tolist()

# Aggregation function to make a list of tags in each row.
agg_function_tag_only = lambda df: [t for t in df['tag'].values.tolist()]
        
# List of tags_tokens.
tag_sequence = grouped.apply(agg_function_tag_only).tolist()

# DataFrame for sentences and tags.
df_sent_tags = pd.DataFrame({'sentences' : sentences, 'tag_sequence' : tag_sequence})

# Remove rows for len less than 3.
id_del = [ii for ii, item in enumerate(sentences) if len(item) < 3]
df_sent_tags.drop(df_sent_tags.index[id_del], inplace=True)

print("Number of sentences having less than 3 words (that is removed): ", len(sentences) - len(df_sent_tags))

# Get the statistics of the number of words per sentence.
# =============================================================================
# lens = pd.Series(list(map(len, df_sent_tags['sentences'].values)))
# print(lens.describe())
# 
# count    4359.000000
# mean       14.588208
# std         8.674760
# min         3.000000
# 25%         9.000000
# 50%        13.000000
# 75%        18.000000
# max        93.000000
# =============================================================================

# Mappings from word/tag to its IDs.
word2idx = {word : ii for ii, word in enumerate(words, 1)}
tag2idx = {tag : ii for ii, tag in enumerate(tags, 0)}
tag2idx['<pad>'] = 17

# Inverse mapping.
idx2word = {ii : word for ii, word in enumerate(words, 1)}
idx2tag = {ii : tag for ii, tag in tag2idx.items()}
idx2tag[17] = '<pad>'

# Pad Index for word_sequence and tag_sequence.
word_pad_id = word2idx['<PAD>']
tag_pad_id = tag2idx['<pad>']

# Unpadded sequences.
# List of token IDs for wach sentence and the tagset. 
df_sent_tags['word_id'] = df_sent_tags['sentences'].apply(lambda x: torch.tensor(utils.word_list2id_list(x, word2idx, tag2idx, mapper = 'word'))) 
df_sent_tags['seq_id'] = df_sent_tags['tag_sequence'].apply(lambda x: torch.tensor(utils.word_list2id_list(x, word2idx, tag2idx, mapper = 'tag'))) 

# Padding the sentences with the ID for <PAD> and tags with the ID for <pad>.
# Maximum length for padding is 60. (Check the KDE plot for the lens variable.)
# If length of a tensor exceeds 60, it'll be post-truncated.
MAX_LEN = 60

# x.
df_sent_tags['padded_word_id'] = df_sent_tags['word_id'].apply(lambda x: utils.padder_func(x, word_pad_id))
# y.
df_sent_tags['padded_seq_id'] = df_sent_tags['seq_id'].apply(lambda x: utils.padder_func(x, tag_pad_id))

# Train - Test split.
x_train, x_test, y_train, y_test = train_test_split(df_sent_tags['padded_word_id'].tolist(),
                                                    df_sent_tags['padded_seq_id'].tolist(), 
                                                    test_size=0.2, 
                                                    random_state=1)

# Configuration.
config = Configuration(bidirection = True, 
                       batch_size = 32,
                       num_layers = 2,
                       hidden_dim = 100, 
                       embedding_dim = 300, 
                       dropout_prob = 0.5,
                       pretrained = False,
                       train_type = (2,3))

# Data Loader.
data_loader = DataLoaderConstant(vocab_size = len(words),
                                  tagset_size = len(tags),
                                  weights = None)

# Creating training and testing iterators.
train_loader = DataLoader(TABSADataset(x_train, y_train, word_pad_id), batch_size = config.batch_size, shuffle=True)
test_loader = DataLoader(TABSADataset(x_test, y_test, word_pad_id), batch_size = config.batch_size, shuffle=True)

#################### MODEL BUILDING ####################

# Model Architecture.
model = BiLSTM(config, data_loader, MAX_LEN)

# Defining the parameters of the network.
optimizer = Adam(model.parameters(), lr = 0.001)
 
# Defining early stopping object.
early_stopping = EarlyStopping(patience = 3, 
                               verbose = True, 
                               delta = 0.0001, 
                               path = os.path.join(os.path.dirname(__file__), r'checkpoints', r'checkpoint.pt'))

# Training Phase.
dynamic_model = DynamicModel(model, optimizer, tag_pad_id)

history = dynamic_model.fit(train_loader, 
                            test_loader, 
                            epochs = 2, 
                            n = 50,               # Calculates the loss and accuracy of the validation set after 50 iterations in an epoch.
                            early_stopping_callback = early_stopping, 
                            return_cache = True, 
                            plot_history = True)