# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 08:25:51 2021

@author: Sagun Shakya
"""
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.model_selection import train_test_split
import os

#LOCAL MODULES.
import utils
from utils import log_sum_exp, argmax
from bilstm_crf_model import BiLSTM_CRF
from dataset_module import TABSADataset, TrainTest
from load_data import get_file_in_df

# Relative path.
filename = os.path.join(os.path.dirname(__file__), r'data', r'data_aspect_target_extraction.txt')

# Parse the txt file in a Pandas DataFrame.
data = get_file_in_df(filename, mode = 'aspect')

# Removing the last 4 rows.
data = data.iloc[:-4]

# Getting the input and output to the model.
xy = TrainTest(data)
words = xy.words
words = ['<PAD>'] + words

# WOrd to index. <PAD> is indexed as 0.
word2idx = {word : ii for ii, word in enumerate(words, 0)}

# Retrieve sentence and tags dataframe.
df_sent_tags = xy.df_sent_tags

# Adding the <pad> to the tag set.
tags = xy.tags

# Add padding.
PAD = '<pad>'
tags = [PAD] + tags

# Tag to Index.
tag2idx = {tag : ii for ii, tag in enumerate(tags, 1)} 

# Word Pad ID and Tag Pad ID.
word_pad_id = word2idx['<PAD>']
tag_pad_id = tag2idx[PAD]

# Inverse mapping.
dict_flipper = lambda mydict : {v : k for k,v in mydict.items()}
idx2word = dict_flipper(word2idx)
idx2tag = dict_flipper(tag2idx)

#%% Unpadded sequences.
# List of token IDs for wach sentence and the tagset. 
df_sent_tags['word_id'] = df_sent_tags['sentences'].apply(lambda x: torch.tensor(utils.word_list2id_list(x, 
                                                                                                        word2idx, 
                                                                                                        tag2idx, 
                                                                                                        mapper = 'word'))) 
df_sent_tags['seq_id'] = df_sent_tags['tag_sequence'].apply(lambda x: torch.tensor(utils.word_list2id_list(x, 
                                                                                                          word2idx, 
                                                                                                          tag2idx, 
                                                                                                          mapper = 'tag'))) 


# Add padding.
# x.
df_sent_tags['padded_word_id'] = df_sent_tags['word_id'].apply(lambda x: utils.padder_func(x, word_pad_id, max_len = 60))
# y.
df_sent_tags['padded_seq_id'] = df_sent_tags['seq_id'].apply(lambda x: utils.padder_func(x, tag_pad_id, max_len = 60))





#%% Prepare data.
x = df_sent_tags['padded_word_id'].tolist()
y = df_sent_tags['padded_seq_id'].tolist()

# Converting to torch tensor.
x_new = utils.list_to_tensor(x)
y_new = utils.list_to_tensor(y)   

# Train - Test split.
x_train, x_test, y_train, y_test = train_test_split(x_new, 
                                                    y_new, 
                                                    test_size = 0.2, 
                                                    random_state = 1)

#%% Constants. 
# TO-DO: Needs to be put in a config file.
tag_names = tags

tagset_size = len(tag_names)
vocab_size = len(words)
max_len = 60
batch_size = 32
hidden_dim = 128
embedding_dim = 300
num_layers = 2
dropout_prob = 0

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_names = [START_TAG] + tag_names + [STOP_TAG]

tag2idx[START_TAG] = 0     # 10
tag2idx[STOP_TAG] = len(tag2idx)      # 11
idx2tag[0] = START_TAG
idx2tag[tag2idx[STOP_TAG]] = STOP_TAG

# Updated number of tags.
num_tags = len(tag_names) + 2

# For the model.
LEARNING_WEIGHT = 1e-2
WEIGHT_DECAY = 1e-4

# Number of epochs.
epochs = 5
#%% Data Iterators: Train and Test.
train_loader = DataLoader(TABSADataset(x_train, y_train, word_pad_id), batch_size = batch_size, shuffle=True)
test_loader = DataLoader(TABSADataset(x_test, y_test, word_pad_id), batch_size = batch_size, shuffle=True)

#%% Model  build.
model = BiLSTM_CRF(batch_size, 
                   max_len, 
                   embedding_dim, 
                   hidden_dim, 
                   num_layers, 
                   True, 
                   dropout_prob, 
                   vocab_size, 
                   num_tags, 
                   tag2idx, 
                   START_TAG, 
                   STOP_TAG)

# Defining the parameters of the network.
optimizer = Adam(model.parameters(), lr = LEARNING_WEIGHT, weight_decay = WEIGHT_DECAY)

#%% Train.
loss_cache_train = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}
loss_cache_train_grained = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}

for ee in range(epochs):
    loss_train = []
    running_loss = 0
    for ii, ((seq, seq_len), tags) in enumerate(train_loader):
        
        model.zero_grad()
        
# =============================================================================
#         print("Seq shape: ", seq.shape)
#         print("Tags shape: ", tags.shape)
#         print("Num Sequences: ", len(seq))
# =============================================================================
    
        criterion = model.forward(seq, seq_len, tags)
        
        mean_batch_loss =  criterion.mean().item()    
        running_loss += mean_batch_loss   # Computes the mean of NLL loss for the batch (all sentences).
        
        loss_train.append(criterion)
        
        criterion.backward()       
        optimizer.step()
        
        if (ii + 1) % 25 == 0:
            # Verbose.
            print(f"Training Loss Till:\nEpoch : {ee + 1} of {epochs} || Iteration : {ii + 1} || Training Loss : {mean_batch_loss}")
            
            # Make predictions using viterbi decoder & Capture validation accuracy.
            val_accuracy = []
            for jj, ((seqv, seq_lenv), tagsv) in enumerate(test_loader):
                
                model.eval()
                with torch.no_grad():
                    best_path_scores, best_path = model.decode(seqv, seq_lenv, tagsv)
                    best_path = torch.tensor(best_path)
                    
                    # Find accuracy for the batch.
                    cat_accuracy = utils.categorical_accuracy(best_path, seqv, seq_lenv)
                    val_accuracy.append(cat_accuracy)
                
            print(f"\nAverage validation accuracy: {utils.compute_average(val_accuracy)}")
            print("Moving on...")
    
    loss_cache_train_grained['epoch_' + str(ee+1)] = loss_train
    loss_cache_train['epoch_' + str(ee+1)] = running_loss / (ii + 1)
    
# =============================================================================
# #%% Exp
# import torch.nn as nn
# 
# for ii, ((seq, seq_len), tags) in enumerate(train_loader):
#     lstm_feats = model._lstm_features(sentence = seq, seq_len = seq_len, tag = tags)
#     break
# 
# #%% crf viterbi
# best_path_scores, best_path = model.decode(seq, seq_len, tags)
# =============================================================================
