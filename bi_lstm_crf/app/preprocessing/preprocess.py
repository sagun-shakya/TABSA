"""
Date: 02-01-022
Author: Sagun Shakya

Preprocessor class. 

This class implements the following tasks: 
    - Read the raw data located at dir(sample_corpus).
        - The data required are named as {dataset1.pkl, vocab.json, tags.json}.
        - dataset1.pkl is a dataframe with columns [sentences, tag_sequences].
        - The x values are list of tokens and the y values are the list of tokens.
    - Convert the list of tokens to one hot vectors.
        - OOV_IDX = len(vocab)
    - Fix the length of the sequences to max_seq_len using padder_func.
        - PAD_IDX = 0
    - Train-Test-Validation Split.
"""

# Importing necessary libraries.
from os.path import join, exists
import numpy as np
from tqdm import tqdm
import torch
from bi_lstm_crf.app.preprocessing.utils import save_json_file, load_json_file
from bi_lstm_crf.app.preprocessing.utils import wordlist_to_idx, padder_func
import pandas as pd
import json

# Constants.
FILE_VOCAB = "vocab.json"
FILE_TAGS = "tags.json"
FILE_DATASET = "dataset1.pkl"
FILE_DATASET_CACHE = "dataset_cache_{}.npz"

START_TAG = "<START>"
STOP_TAG = "<STOP>"

PAD = "<PAD>"
OOV = "<OOV>"

class Preprocessor:
    '''
    Takes in the raw data in the form of a pickled dataframe and provides the input for train function.
    
    Parameters:
    config_dir -- path to corpus_dir.
    save_config_dir -- path to model_dir to save the models and cache.
    verbose -- display information. 
    '''

    def __init__(self, config_dir, save_config_dir=None, verbose=True):
        self.config_dir = config_dir
        self.verbose = verbose

        self.vocab, self.vocab_dict = self.__load_list_file(FILE_VOCAB, offset=1, verbose=verbose)
        self.tags, self.tags_dict = self.__load_list_file(FILE_TAGS, verbose=verbose)
        if save_config_dir:
            self.__save_config(save_config_dir)

        self.PAD_IDX = 0
        self.OOV_IDX = len(self.vocab)
        self.__adjust_vocab()

    def __load_list_file(self, file_name, offset=0, verbose=False):
        file_path = join(self.config_dir, file_name)
        if not exists(file_path):
            raise ValueError('"{}" file does not exist.'.format(file_path))
        else:
            elements = load_json_file(file_path)
            elements_dict = {w: idx + offset for idx, w in enumerate(elements)}
            if verbose:
                print("config {} loaded".format(file_path))
            return elements, elements_dict

    def __adjust_vocab(self):
        '''
        Insert PAD and OOV tokens in the vocabulary list.
        '''
        self.vocab.insert(0, PAD)
        self.vocab_dict[PAD] = 0

        self.vocab.append(OOV)
        self.vocab_dict[OOV] = len(self.vocab) - 1

    def __save_config(self, dst_dir):
        char_file = join(dst_dir, FILE_VOCAB)
        save_json_file(self.vocab, char_file)

        tag_file = join(dst_dir, FILE_TAGS)
        save_json_file(self.tags, tag_file)

        if self.verbose:
            print("tag dict file => {}".format(tag_file))
            print("tag dict file => {}".format(char_file))

    @staticmethod
    def __cache_file_path(corpus_dir, max_seq_len):
        return join(corpus_dir, FILE_DATASET_CACHE.format(max_seq_len))

    def load_dataset(self, corpus_dir, val_split, test_split, max_seq_len):
        """
        Loads the train set.

        :return: (xs, ys)
            xs: [B, L]
            ys: [B, L, C]
        """        
        xs, ys = self.build_corpus(corpus_dir, max_seq_len)
        # split the dataset
        total_count = len(xs)
        assert total_count == len(ys)
        val_count = int(total_count * val_split)
        test_count = int(total_count * test_split)
        train_count = total_count - val_count - test_count
        assert train_count > 0 and val_count > 0

        indices = np.cumsum([0, train_count, val_count, test_count])
        datasets = [(xs[s:e], ys[s:e]) for s, e in zip(indices[:-1], indices[1:])]
        print("datasets loaded:")
        for (xs_, ys_), name in zip(datasets, ["train", "val", "test"]):
            print("\t{}: {}, {}".format(name, xs_.shape, ys_.shape))
        return datasets

    def decode_tags(self, batch_tags):
        batch_tags = [
            [self.tags[t] for t in tags]
            for tags in batch_tags
        ]
        return batch_tags


    def build_corpus(self, corpus_dir, max_seq_len):
        '''
        Reads the raw data from dataset1.pkl.

        Parameters:
        corpus_dir -- Path to sample_corpus.
        max_seq_len -- Length of the input vectors (includes padding).

        Returns:
        Tuple. Torch tensors of X values representing sentences in one-hot-vector form and
        Y values representing tags in one-hot-vector form including padding element. 
        '''
        
        file_path = join(corpus_dir, FILE_DATASET)
        
        f = pd.read_pickle(file_path)
        
        xs = f['sentences'].apply(lambda x: padder_func(wordlist_to_idx(x, self.vocab_dict, self.OOV_IDX), 
                                                        value = self.PAD_IDX, 
                                                        max_len = max_seq_len))
        ys = f['tag_sequence'].apply(lambda x: padder_func(wordlist_to_idx(x, self.tags_dict, self.OOV_IDX), 
                                                           value = self.PAD_IDX, 
                                                           max_len = max_seq_len))
        
        xs = xs.to_list()
        ys = ys.to_list()
        
        # save train set
        cache_file = self.__cache_file_path(corpus_dir, max_seq_len)
        np.savez(cache_file, xs=xs, ys=ys)
        print("dataset cache({}, {}) => {}".format(np.asarray(xs).shape, np.asarray(ys).shape, cache_file))
        
        return torch.tensor(xs), torch.tensor(ys)