# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:10:19 2021

@author: Sagun Shakya
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable

import os

#%% Get data from pickle.
import pickle
from sklearn.model_selection import train_test_split
os.chdir(r'D:/ML_projects/TABSA/bisltm_crf_pytorch/data')

# load : get the data from file
file_path = r'df_sent_tags.pkl'
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


#%% Constants.
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


START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_names = [START_TAG] + tag_names + [STOP_TAG]

tag2idx = {tag : ii for ii, tag in enumerate(tag_names)}
idx2tag = {v:k for k,v in tag2idx.items()}

# Updated number of tags.
num_tags = tagset_size + 2

# For the model.
LEARNING_WEIGHT = 5e-2
WEIGHT_DECAY = 1e-4

# Word Pad ID.
word_pad_id = 11625
tag_pad_id = 9

#%% Helper Functions.
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


#%% Model.
class BiLSTM_CRF(nn.Module):
    def __init__(self, 
                 batch_size, 
                 max_len, 
                 embedding_dim,
                 hidden_dim,
                 num_layers,
                 bidirection,
                 dropout_prob,
                 vocab_size,
                 num_tags,
                 tag2idx):

        super(BiLSTM_CRF, self).__init__()
        
        self.bidirection = bidirection
        self.batch_size = batch_size
        self.num_layers =  num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.tag2idx = tag2idx
        
        ## NN Layers.
        
        # Embedding Layer.
        self.embedding = nn.Embedding(num_embeddings = self.vocab_size, 
                                      embedding_dim = self.embedding_dim)
        
        # BiLSTM Layer.
        self.bilstm = nn.LSTM(input_size = self.embedding_dim,  
                                hidden_size = self.hidden_dim//2,
                                num_layers = self.num_layers, 
                                bidirectional = self.bidirection,
                                batch_first = True)
        
        # Dense Layer.
        self.fc = nn.Linear(in_features = self.hidden_dim, 
                       out_features = self.num_tags)
        
        ## Transition Matrix (Learnable). 
        # T_(ij) == T [j --> i].
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags))
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag.
        self.transitions.data[self.tag2idx[START_TAG], :] = -10000
        self.transitions.data[:, self.tag2idx[STOP_TAG]] = -10000
        
    def _forward_alg(self, feats):
        """Core magic of the Conditional Random Field.  
        
        Input:
            The feature from LSTM layer.
        
        Since we’re using PyTorch to compute gradients for us, 
        we technically only need the forward part of the forward-backward 
        algorithm """
    
        # Do the forward algorithm to compute the partition function.
        init_alphas = torch.full((1, self.num_tags), -10000.)
    
        # START_TAG ("<START>") has all of the score.
        init_alphas[0][self.tag2idx[START_TAG]] = 0.
        forward_var = init_alphas
    
        # Iterate through the sentence.
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep.
            
            for next_tag in range(self.num_tags):
                # Broadcast the emission score: it is the same regardless of the previous tag.
                emit_score = feat[next_tag].view(1, -1).expand(1, self.num_tags)
    
                # the ith entry of trans_score is the score of transitioning to next_tag from i.
                trans_score = self.transitions[next_tag].view(1, -1)
                                                              
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp.
                next_tag_var = forward_var + trans_score + emit_score
    
                # The forward variable for this tag is log-sum-exp of all the scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
    
            forward_var = torch.cat(alphas_t).view(1, -1)
    
        terminal_var = forward_var + self.transitions[self.tag2idx[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        """Gives the score of a provided tag sequence"""

        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag2idx[START_TAG]], dtype=torch.long), 
                          tags])
        
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
            
        score = score + self.transitions[self.tag2idx[STOP_TAG], tags[-1]]
        return score
    
    def neg_log_likelihood(self, feats, tags):
        """Calculate the negative log likelihood given a sequence and labels.
        This is used in training (only) because we don't need to create
        and check the B-I-O tags themselves - only the score is important
        here for calculating the loss."""
        
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        nll = forward_score - gold_score
        return nll
    
    def _get_list_average(self, mylist):
        return sum(mylist) / len(mylist)
    
    def forward(self, sentence, seq_len, tag):
        '''
        Forward pass in training.

        Parameters
        ----------
        sentence : Torch tensor.
            DESCRIPTION.
        seq_len : Torch tensor.
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Get word embeddings in a sentence.
        sent_embedding = self.embedding(sentence)
        
        # Packing the output of the embedding layer.
        packed_input = pack_padded_sequence(sent_embedding, 
                                            lengths = seq_len.clamp(max = 192), 
                                            batch_first = True, 
                                            enforce_sorted = False)
        
        # BiLSTM layer.
        packed_output, (h_t, c_t) = self.bilstm(packed_input)

        # Inverting the packing operation.
        out, input_sizes = pad_packed_sequence(packed_output, batch_first = True, total_length = self.max_len)

        # Linear layer.
        linear = self.fc(out)
        
        # loss.
        nll_loss = Variable(self._get_list_average([self.neg_log_likelihood(sent, tag) for sent, tag in zip(linear, tag)]), requires_grad = True)     
        
        return nll_loss

        
    
    
    

#%% DataLoader Class.
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

#%% Create data iterators.

train_loader = DataLoader(TABSADataset(x_train, y_train, word_pad_id), batch_size = batch_size, shuffle=True)
test_loader = DataLoader(TABSADataset(x_test, y_test, word_pad_id), batch_size = batch_size, shuffle=True)
    
#%% Initialize model.
model = BiLSTM_CRF(batch_size, max_len, 300, hidden_dim, 1, True, 0.5, vocab_size, num_tags, tag2idx)        

# Defining the parameters of the network.
optimizer = Adam(model.parameters(), lr = 0.001)
 
# =============================================================================
# # Defining early stopping object.
# if not os.path.exists(os.path.join(os.path.dirname(__file__), r'checkpoints')):
#     os.mkdir(os.path.join(os.path.dirname(__file__), r'checkpoints'))
# 
# checkpoint_filepath = os.path.join(os.path.dirname(__file__), r'checkpoints', r'checkpoint.pt')
#     
# early_stopping = EarlyStopping(patience = 3, 
#                                verbose = True, 
#                                delta = 0.0001, 
#                                path = checkpoint_filepath)
# =============================================================================

#%% Experiment.
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

#%% Experiment.
def generate_sample(x_train, y_train):
    sample = x_train[:]
    replace_sample = torch.full(size = (len(sample), 60), fill_value = 0)
    for ii in range(len(sample)):
        replace_sample[ii] = sample[ii]
    
    tags_sample = y_train[:]    
    return replace_sample, tags_sample

sample, tags_sample = generate_sample(x_train, y_train)

for ii, ((seq, seq_len), tags) in enumerate(DataLoader(TABSADataset(sample, tags_sample, word_pad_id), batch_size = batch_size, shuffle=True)):
    print("Iter: ", ii + 1)
    print("\nSample Shape", seq.shape)
    print(seq)
    print("\nTags Shape", tags.shape)  
    print(tags.shape)
    out = model.forward(seq, seq_len, tags)
    print("\nOut Shape: ", out.shape) 
    print() 
    if ii == 5:
        break
    
#%% Train.
epochs = 5

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
        
        print('Epoch : ', ee+1)
        model.zero_grad()
        
        print("Seq shape: ", seq.shape)
        print("Tags shape: ", tags.shape)
        print("Num Sequences: ", len(seq))
        
        emb = nn.Embedding(vocab_size, embedding_dim)(seq)
    
        criterion = model.forward(seq, seq_len, tags)
        
        mean_batch_loss =  criterion.mean().item()    
        running_loss += mean_batch_loss   # Computes the mean of NLL loss for the batch (all sentences).
        
        loss_train.append(criterion)
        criterion.backward()
        
        optimizer.step()
        
    loss_cache_train_grained['epoch_' + str(ee+1)] = loss_train
    loss_cache_train['epoch_' + str(ee+1)] = running_loss / (ii + 1)
















