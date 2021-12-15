# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 07:46:45 2021

@author: Sagun Shakya
"""

# Importing necessary libraries.
import numpy as np
import pandas as pd


def get_file_in_df(filename, mode = 'aspect'):
    '''
    Gets the file in a DataFrame.
    
     Parameters:
         filename -- Path to the file.
         
     Returns:
         DataFrame. Columns = ['sentence_no', 'token', 'start', 'end', 'tag']
    '''
    
    with open(filename, encoding = 'utf-8') as ff:
        text = ff.readlines()
        ff.close()
    
    
    text = [line.split() for line in text]
    
    for ii, line in enumerate(text):        
        if line == []:
            text[ii] = ['_'] * 4
            
    df = pd.DataFrame(text, columns = ['token', 'start', 'end', 'tag'])
    
    sentence_no = ['_']*len(df)
    
    count = 1
    for ii in range(len(df)):
        sentence_no[ii] = 'sentence:' + str(count)
        if df['start'].iloc[ii] == '_':
            sentence_no[ii] = '_'
            count += 1
            
    sentence_no = pd.DataFrame({'sentence_no': sentence_no})
    
    df = pd.concat([sentence_no, df], axis = 1).replace('_', np.nan)
    df.dropna(inplace = True)
    
    aspect_categories = ['B-GENERAL', 'I-GENERAL', 
                         'B-PROFANITY', 'I-PROFANITY', 
                         'B-VIOLENCE', 'I-VIOLENCE', 
                         'B-FEEDBACK', 'I-FEEDBACK']
    
    target_categories = ['B-PER', 'I-PER',  
                         'B-ORG', 'I-ORG', 
                         'B-LOC', 'I-LOC', 
                         'B-MISC', 'I-MISC']
    
    # Function to remove aspect/target categories from data['tag'].
    remove_function = lambda series, category: series.replace(category, ['O']*len(category)) 
    
    if mode == 'aspect' or mode == 'ASPECT':
        # Removing target categories from the set of tags so that we are left with aspect categories only.
        df['tag'] = remove_function(df['tag'], target_categories) 
        
    elif mode == 'target' or mode == 'TARGET':
        # Removing aspect categories from the set of tags so that we are left with target categories only.
        df['tag'] = remove_function(df['tag'], aspect_categories)
        
    elif mode == 'combined' or mode == 'COMBINED':
        pass
    
    return df        