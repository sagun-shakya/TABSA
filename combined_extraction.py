# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 09:47:40 2021

@author: Sagun Shakya

DESCRIPTION:
    The main task is to label each token to one of the following TARGET CATEGORIES.
    - ('B-PER', 'I-PER', 
       'B-ORG', 'I-ORG', 
       'B-LOC', 'I-LOC', 
       'B-MISC', 'I-MISC',
       'B-FEEDBACK',
        'B-GENERAL',
        'B-PROFANITY',
        'B-VIOLENCE',
        'I-FEEDBACK',
        'I-GENERAL',
        'I-PROFANITY',
        'I-VIOLENCE',
        'O'             )

"""

#%% Importing necessary libraries.

import os

from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.model_selection import train_test_split

# LOCAL MODULES.
from load_data import get_file_in_df
from dataset_module import TABSADataset, TrainTest
from config import Configuration
from dataloader import DataLoaderConstant
from BiLSTM_model import BiLSTM
from dynamic_model import DynamicModel
from EarlyStopping_module import EarlyStopping

#%% Input and Output Variables.

# Constants.
BATCH_SIZE = 32

# Relative path.
filename = os.path.join(os.path.dirname(__file__), r'data', r'data_aspect_target_extraction.txt')

# Parse the txt file in a Pandas DataFrame.
data = get_file_in_df(filename, mode = 'combined')

# Removing the last 4 rows.
data = data.iloc[:-4]

# Getting the input and output to the model.
xy = TrainTest(data)
df_sent_tags = xy.df_sent_tags

# Fix length of the input tensors. Default 60.
MAX_LEN = len(df_sent_tags['padded_seq_id'].iloc[0])

# to_remove.
#to_remove = {'...', '!', ',', '||', '/', '.', '(', ')', '..', '!', '@', '#', '%', '।'}

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
data_loader = DataLoaderConstant(vocab_size = len(xy.words),
                                 tagset_size = len(xy.tags),
                                 weights = None)

# Creating training and testing iterators.
train_loader = DataLoader(TABSADataset(x_train, y_train, xy.word_pad_id), batch_size = config.batch_size, shuffle=True)
test_loader = DataLoader(TABSADataset(x_test, y_test, xy.word_pad_id), batch_size = config.batch_size, shuffle=True)


#%%  MODEL BUILDING

# Model Architecture.
model = BiLSTM(config, data_loader, MAX_LEN)

# Defining the parameters of the network.
optimizer = Adam(model.parameters(), lr = 0.001)
 
# Defining early stopping object.
early_stopping = EarlyStopping(patience = 3, 
                               verbose = True, 
                               delta = 0.0001, 
                               path = os.path.join(os.path.dirname(__file__), r'checkpoints', r'combined', r'checkpoint.pt'))

# Training Phase.
dynamic_model = DynamicModel(model, optimizer, xy.tag_pad_id)

history = dynamic_model.fit(train_loader, 
                            test_loader, 
                            epochs = 2, 
                            n = 50,               # Calculates the loss and accuracy of the validation set after 50 iterations in an epoch.
                            early_stopping_callback = early_stopping, 
                            return_cache = True, 
                            plot_history = True)

