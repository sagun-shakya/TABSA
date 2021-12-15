# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 08:15:40 2021

@author: Sagun Shakya
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

# LOACL MODULES.
from utils import log_sum_exp, argmax

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
                 tag2idx, 
                 START_TAG, 
                 STOP_TAG):

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
        
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"

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
        self.transitions.data[self.tag2idx[self.START_TAG], :] = -10000
        self.transitions.data[:, self.tag2idx[self.STOP_TAG]] = -10000
        
    def _forward_alg(self, feats):
        """Core magic of the Conditional Random Field.  
        
        Input:
            The feature from LSTM layer.
        
        Since we’re using PyTorch to compute gradients for us, 
        we technically only need the forward part of the forward-backward 
        algorithm """
    
        # Do the forward algorithm to compute the partition function.
        init_alphas = torch.full((1, self.num_tags), -10000.)
    
        # self.START_TAG ("<START>") has all of the score.
        init_alphas[0][self.tag2idx[self.START_TAG]] = 0.
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
    
        terminal_var = forward_var + self.transitions[self.tag2idx[self.STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        """Gives the score of a provided tag sequence"""

        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag2idx[self.START_TAG]], dtype=torch.long), 
                          tags])
        
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
            
        score = score + self.transitions[self.tag2idx[self.STOP_TAG], tags[-1]]
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
    
    def _lstm_features(self, sentence, seq_len, tag):
        """Gets the LSTM features for a batch."""
        
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
        return linear
    
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
    
    def _viterbi_decode(self, feats):
        """Implements Viterbi algorithm for finding most likely sequence of labels.
        Used in the forward pass of the network.
    
        We take the maximum over the previous states as opposed to the sum. 
        Input:
            loglikelihoods: torch tensor.
        Output:
            tuple. The first entry is the loglikelihood of this sequence. The second is 
            the most likely sequence of labels. 
        """
        backpointers = []
    
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.num_tags), -10000.)
        init_vvars[0][self.tag2idx[self.START_TAG]] = 0
    
        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step
    
            for next_tag in range(self.num_tags):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
           
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
    
        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2idx[self.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
    
        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
            
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag2idx[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path
    
    def decode(self, sentence, seq_len, tag):
        '''Apply Viterbi algorithm to generate best tags.
        Used for validation only.'''
        
        lstm_feature = self._lstm_features(sentence, seq_len, tag)    # Shape (batch_size, max_len, num_tags)
        best_path_scores = []
        best_path = []
        
        for feature in lstm_feature:
            path_score_sent, path_sent = self._viterbi_decode(feature)
            best_path_scores.append(path_score_sent)
            best_path.append(path_sent)
        return best_path_scores, best_path
            
