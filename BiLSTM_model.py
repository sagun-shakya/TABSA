# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 08:46:26 2021

@author: Sagun Shakya
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):
    def __init__(self, config, data_loader, max_len):
        super(BiLSTM, self).__init__()
        self.config = config
        self.data_loader = data_loader
        self.max_len = max_len
        
        # From config.
        self.bidirection = config.bidirection
        self.batch_size = config.batch_size
        self.num_layers =  config.num_layers
        self.hidden_dim = config.hidden_dim
        self.embedding_dim = config.embedding_dim
        self.dropout_prob = config.dropout_prob
        self.train_type = config.train_type
        self.pretrained = config.pretrained
        
        # From Dataloader.
        self.vocab_size = data_loader.vocab_size
        self.tagset_size = data_loader.tagset_size
        self.weights = data_loader.weights
        
        if self.pretrained:
            self.embedding = nn.Embedding.from_pretrained(data_loader.weights)      # TO-DO.
        else:
            self.embedding = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = self.embedding_dim) 
            
# =============================================================================
#         self.ac_size = data_loader.ac_size
#         self.ac_embeddings = nn.Embedding(self.ac_size, self.embedding_dim)
# =============================================================================
        
        self.bilstm = nn.LSTM(input_size = self.embedding_dim,  
                              hidden_size = self.hidden_dim,
                              num_layers = self.num_layers, 
                              bidirectional = self.bidirection,
                              batch_first = True)
        
        self.linear = nn.Linear(self.hidden_dim*2, self.tagset_size)
        self.dropout = nn.Dropout(self.dropout_prob)

            
    def forward(self, sentence, seq_len):
        # Length of each sequence.
        #seq_len = torch.LongTensor(list(map(len, sentence)))
        
        # Embedding layer.
        sent_embeddings = self.embedding(sentence)

        # Adding dropout to the embedding output.
        sent_embeddings = self.dropout(sent_embeddings)

        # Packing the output of the embedding layer.
        packed_input = pack_padded_sequence(sent_embeddings, 
                                            lengths = seq_len.clamp(max = 192), 
                                            batch_first = True, 
                                            enforce_sorted = False)
        
# =============================================================================
#         diff = packed_input.batch_sizes.sum().item() - packed_input.data.shape[0]
#         if diff > 0:
#             print("\nShape mismatch found.")
#             print("Sum of seq_len: ", packed_input.batch_sizes.sum().item())
#             print("Shape of packed input: ", packed_input.data.shape)
#             print("Skipping...")
#             return None
# =============================================================================
        
        # BiLSTM layer.
        packed_output, (h_t, c_t) = self.bilstm(packed_input)

        # Inverting the packing operation.
        out, input_sizes = pad_packed_sequence(packed_output, batch_first = True, total_length = self.max_len)

        #Linear layer.
        output = self.linear(out)

        # Softmax. Then, logarithmic transform.
        pred_prob = F.log_softmax(output, dim=1)

        # Transposing the dimension: (batch_size, max_len, num_tags) --> (batch_size, num_tags, max_len).
        pred_prob = pred_prob.permute(0, 2, 1) 

        return pred_prob
        
        

        
    
        
# =============================================================================
#     def forward(self, text, at, ac):
#         #text = [batch size, sent len]
#         text = text.permute(1, 0)
#         at = at.permute(1, 0)
#         ac = ac.permute(1, 0)        
#         
#         embedded = self.embedding(text)
#         at_emb = self.embedding(at)
#         ac_emb = self.ac_embeddings(ac)
#         
#         embedded = torch.cat([embedded, ac_emb], dim=0)    
#         
#         # Only concatenate text and aspect term
#         if self.train_type in [2,3]:
#             embedded = torch.cat((embedded, at_emb), dim=0)
#     
#         #embedded = [sent len (embedded+aspect_emb), batch size, emb dim]
#         
#         embedded, (hidden, cell) = self.bilstm(self.dropout(embedded))
#                 
#         # embedded = [batch size, sent_len, num_dim * hidden_dim]
#         # hidden = [num_dim, sent_len, hidden_dim]
#     
#         hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
#     
#         # hidden = [batch size, hid dim * num directions]
#         final = F.softmax(self.fc(hidden), dim=-1)
#         
#         return final
# =============================================================================
        
        
        
        
        
        