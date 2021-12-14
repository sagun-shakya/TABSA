# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 08:49:47 2021

@author: Sagun Shakya
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os

#%% Get data from pickle.
import pickle
from sklearn.model_selection import train_test_split
os.chdir(r'D:/ML_projects/TABSA/bisltm_crf_pytorch/data')

# load : get the data from file
file_path = r'df_sent_tags.pkl'
df_sent_tags = pickle.load(open(file_path, "rb"))

# Train - Test split.
x_train, x_test, y_train, y_test = train_test_split(df_sent_tags['padded_word_id'].tolist(),
                                                    df_sent_tags['padded_seq_id'].tolist(), 
                                                    test_size=0.2, 
                                                    random_state=1)

#%% Constants.
tag_names = ['B-FEEDBACK',
            'B-GENERAL',
            'B-PROFANITY',
            'B-VIOLENCE',
            'I-FEEDBACK',
            'I-GENERAL',
            'I-PROFANITY',
            'I-VIOLENCE',
            'O']
tagset_size = len(tag_names)
vocab_size = 11626
batch_size = 32
hidden_dim = 128


START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_names = [START_TAG] + tag_names + [STOP_TAG]

tag2idx = {tag : ii for ii, tag in enumerate(tag_names)}
idx2tag = {v:k for k,v in tag2idx.items()}

# Updated number of tags.
num_tags = tagset_size + 2

# T_(ij) == T [j --> i].
transitions = nn.Parameter(torch.randn(num_tags, num_tags))


#%% Functions.
def argmax(vec):
    """Return the argmax as a python int"""
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    """Compute log sum exp in a numerically stable way for 
    the forward algorithm."""
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    result = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
    return result

def _forward_alg(feats):
    """Core magic of the Conditional Random Field.  
    
    Input:
        The feature from LSTM layer.
    
    Since we’re using PyTorch to compute gradients for us, 
    we technically only need the forward part of the forward-backward 
    algorithm """

    # Do the forward algorithm to compute the partition function.
    init_alphas = torch.full((1, num_tags), -10000.)

    # START_TAG ("<START>") has all of the score.
    init_alphas[0][tag2idx[START_TAG]] = 0.
    forward_var = init_alphas

    # Iterate through the sentence.
    for feat in feats:
        alphas_t = []  # The forward tensors at this timestep.
        
        for next_tag in range(num_tags):
            # Broadcast the emission score: it is the same regardless of the previous tag.
            emit_score = feat[next_tag].view(1, -1).expand(1, num_tags)

            # the ith entry of trans_score is the score of transitioning to next_tag from i.
            trans_score = transitions[next_tag].view(1, -1)
                                                          
            # The ith entry of next_tag_var is the value for the
            # edge (i -> next_tag) before we do log-sum-exp.
            next_tag_var = forward_var + trans_score + emit_score

            # The forward variable for this tag is log-sum-exp of all the scores.
            alphas_t.append(log_sum_exp(next_tag_var).view(1))

        forward_var = torch.cat(alphas_t).view(1, -1)

    terminal_var = forward_var + transitions[tag2idx[STOP_TAG]]
    alpha = log_sum_exp(terminal_var)
    return alpha

def _score_sentence(feats, tags):
    """Gives the score of a provided tag sequence"""
    # Gives the score of a provided tag sequence
    score = torch.zeros(1)
    tags = torch.cat([torch.tensor([tag2idx[START_TAG]], dtype=torch.long), 
                      tags])
    
    for i, feat in enumerate(feats):
        score = score + transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        
    score = score + transitions[tag2idx[STOP_TAG], tags[-1]]
    return score


#%% Sample input.
sample = x_train[:batch_size]
x = torch.full(size = (batch_size, 60), fill_value = 0)
for ii in range(batch_size):
    x[ii] = sample[ii]

tags_sample = y_train[:batch_size]


#%% Forward.
X = torch.randint(0, 100, size = (batch_size, 60))
print('Input Shape: ', X.shape)

embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = 300)(X)
print('Embedding Shape: ', embedding.shape)

bilstm, (_, _) = nn.LSTM(input_size = 300,  
                hidden_size = hidden_dim//2,
                num_layers = 2, 
                bidirectional = True,
                batch_first = True)(embedding)

print('BiLSTM shape: ', bilstm.shape)

fc = nn.Linear(in_features = hidden_dim, out_features = num_tags)(bilstm)
print('Linear Layer Shape: ', fc.shape)

# T_(ij) == T [j --> i].
transitions = nn.Parameter(torch.randn(num_tags, num_tags))
# These two statements enforce the constraint that we never transfer
# to the start tag and we never transfer from the stop tag.
transitions.data[tag2idx[START_TAG], :] = -10000
transitions.data[:, tag2idx[STOP_TAG]] = -10000
print("Transition matrix shape: ", transitions.shape)

# Calculate the forward score for each sentence and put it in a list.
forward_score = torch.tensor([_forward_alg(sent) for sent in fc])
print('Forward Score Shape:', forward_score.shape)

# Calculate the gold scores for each sentence in the batch and put it in a list
gold_score = torch.tensor([_score_sentence(sent, tag) for sent, tag in zip(fc, tags_sample)])
print('Gold Score Shape:', gold_score.shape)

# Calculate the negative log likelihood for sentence.
nll_loss = forward_score - gold_score
print('nll_loss Shape:', nll_loss.shape)


#%% Sample Train.

sample = x_train[:5*batch_size]
sample = torch.full(size = (5*batch_size, 60), fill_value = 0)
for ii in range(5*batch_size):
    x[ii] = sample[ii]

tags_sample = y_train[:5*batch_size]    

for ii, ((sample, seq_len), tags) in enumerate(DataLoader(TABSADataset(sample, tags_sample, word_pad_id), batch_size = batch_size, shuffle=True)):
    print("Iter: ", ii + 1)
    print(tags.shape)    
    out = model.forward(sample, seq_len, tags)
    print(out.shape) 
    print()  