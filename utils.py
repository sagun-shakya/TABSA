# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:16:18 2021

@author: Sagun Shakya
"""

import torch
from torch.nn.functional import pad
import matplotlib.pyplot as plt


def word_list2id_list(word_list, word2idx, tag2idx, mapper = 'word'):
    '''
    For a given list of tokens, the function replaces the tokens with
    their corresponding idx in the vocabulary.
    '''
    try:
        if mapper == 'word':
            return [word2idx[WORD] for WORD in word_list]
        else:
            return [tag2idx[WORD] for WORD in word_list]
    except:
        return []
    
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

def padder_func(tensor_i, value, max_len = 60):
    '''
    Padder function takes in a one dimensional tensor and post-pads it with a designated value.
    
    Parameters:
        tensor_i -- One dimensional tensor.
        value -- The value to be used as the pad.
        max_len -- The final length of the tensors (default value = 60).
    
    Note that if the length of the tensor exceeds max_len, it will be post-truncated so that the final length is max_len.
    
    Returns:
        Torch tensor. Padded tensor of length equal to max_len.
    '''
    return pad(tensor_i, pad = (0, max_len - len(tensor_i)), mode = "constant", value = value)
    

def list_to_tensor(mylist):
    '''
    Converts a list of tensors to tensor.
    All the tensors inside the list should be of the same length.

    Parameters
    ----------
    mylist : list
        Contains tensors all of length 'max_len'.

    Returns
    -------
    tensor of shape (len(mylist), max_len).

    '''
    
    x_new = torch.full(size = (len(mylist), len(mylist[0])), fill_value = 0)
    
    for ii in range(len(mylist)):
        x_new[ii] = mylist[ii]
        
    return x_new
       
def categorical_accuracy(model_output, true_labels, tag_pad_value = 17):
    try:
        predicted_labels = model_output.argmax(axis = 1)

        error_msg = 'The shape of the predicted_labels doesnt match with that of true_labels.'
        error_msg += f'\nShape of predicted_labels: {predicted_labels.shape}'
        error_msg += f'\nShape of true_labels: {true_labels.shape}'
        assert predicted_labels.shape == true_labels.shape, error_msg
        
        # Mask to filter in non-padded elements.
        non_pad_mask = (true_labels != tag_pad_value)

        model_output_smooth = predicted_labels[non_pad_mask]
        true_labels_smooth = true_labels[non_pad_mask]

        assert model_output_smooth.shape == true_labels_smooth.shape, "The shape of the flattened outputs/labels do not match."

        res = model_output_smooth.eq(true_labels_smooth).to(torch.int8)     # Binary value. 1 for match, 0 for no match.
        correct = res.sum()
        total = len(res)                                                    # Sum of Lengths of sequences in the batch.
        accuracy = correct/total
        return round(accuracy.item(), 4)

    except AssertionError as msg:
        print(msg)
        
# Calculating the average loss for the valdation set in this iteration.
compute_average = lambda arr: sum(arr) / len(arr)

def plot_history_object(training_loss_per_epoch_list, 
                 val_loss_per_epoch_list,
                 training_accuracy_per_epoch_list,
                 val_accuracy_per_epoch_list,
                 EPOCHS,
                 figsize = (15,7), 
                 style = 'dark_background'):
    
    plt.figure(figsize = figsize)
    plt.style.use(style)
    
    plt.subplot(1,2,1)
    plt.plot(training_loss_per_epoch_list, color = 'red', label = 'Training Loss')
    plt.plot(val_loss_per_epoch_list, color = 'green', label = 'Validation Loss')
    plt.xticks(tuple(range(1, EPOCHS + 1)))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss curve')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(training_accuracy_per_epoch_list, color = 'red', label = 'Training Accuracy')
    plt.plot(val_accuracy_per_epoch_list, color = 'green', label = 'Validation Accuracy')
    plt.xticks(tuple(range(1, EPOCHS + 1)))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy curve')
    plt.legend()
    
    plt.show()
    return