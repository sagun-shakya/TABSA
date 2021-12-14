# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 08:25:51 2021

@author: Sagun Shakya
"""
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import pickle
from sklearn.model_selection import train_test_split
import os

#LOCAL MODULES.
#from utils import log_sum_exp, argmax
from bilstm_crf_model import BiLSTM_CRF
from dataset_module import TABSADataset

# load data.
file_path = os.path.join(os.path.dirname(__file__), r'data', r'df_sent_tags.pkl')

# load : get the data from file
df_sent_tags = pickle.load(open(file_path, "rb"))

x = df_sent_tags['padded_word_id'].tolist()
y = df_sent_tags['padded_seq_id'].tolist()

x_new = torch.full(size = (len(x), len(x[0])), fill_value = 0)
y_new = torch.full(size = (len(y), len(y[0])), fill_value = 0)

for ii in range(len(x)):
    x_new[ii] = x[ii]
    y_new[ii] = y[ii]
    

# Train - Test split.
x_train, x_test, y_train, y_test = train_test_split(x_new, 
                                                    y_new, 
                                                    test_size = 0.2, 
                                                    random_state = 1)

# Constants. 
# TO-DO: Needs to be put in a config file.
tag_names = ['B-FEEDBACK', 'I-FEEDBACK',
            'B-GENERAL', 'I-GENERAL',
            'B-PROFANITY', 'I-PROFANITY',
            'B-VIOLENCE', 'I-VIOLENCE',
            'O']

tagset_size = len(tag_names)
vocab_size = 11626
max_len = 60
batch_size = 32
hidden_dim = 128
embedding_dim = 300
num_layers = 2
dropout_prob = 0

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_names = [START_TAG] + tag_names + [STOP_TAG]

tag2idx = {tag : ii for ii, tag in enumerate(tag_names)}
idx2tag = {v:k for k,v in tag2idx.items()}

# Updated number of tags.
num_tags = tagset_size + 2

# For the model.
LEARNING_WEIGHT = 1e-2
WEIGHT_DECAY = 1e-4

# Word Pad ID.
word_pad_id = 11625
tag_pad_id = 9

epochs = 5

# Data Iterators: Train and Test.
train_loader = DataLoader(TABSADataset(x_train, y_train, word_pad_id), batch_size = batch_size, shuffle=True)
test_loader = DataLoader(TABSADataset(x_test, y_test, word_pad_id), batch_size = batch_size, shuffle=True)

# Model  build.
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

# Train.

# Cache for accuracy and losses in each epoch for training and validatin sets.
#accuracy_cache_train = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}
#accuracy_cache_val = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}

loss_cache_train = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}
#loss_cache_val = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}

# Fine-grained accuracy and loss cache to see how these values evolve for each batch within each epoch.
#accuracy_cache_train_grained = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}
#accuracy_cache_val_grained = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}

loss_cache_train_grained = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}
#loss_cache_val_grained = {"epoch_" + str(ii + 1) : None for ii in range(epochs)}

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
        
        if (ii + 1) % 50 == 0:
            print(f"Training Loss Till:\nEpoch : {ee + 1} of epochs || Iteration : {ii + 1} || Training Loss : {mean_batch_loss}")
            
    loss_cache_train_grained['epoch_' + str(ee+1)] = loss_train
    loss_cache_train['epoch_' + str(ee+1)] = running_loss / (ii + 1)