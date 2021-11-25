# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:05:52 2021

@author: Sagun Shakya
"""

#Importing necessary libraries.
import os
import pandas as pd
import numpy as np

# LOCAL MODULES.
from load_data import get_file_in_df

# Relative path.
filename = os.path.join(os.path.dirname(__file__), r'data', r'data_aspect_target_extraction.txt')

# Parse the txt file in a Pandas DataFrame.
data = get_file_in_df(filename)

# Removing the last 4 rows.
data = data.iloc[:-4]

# Unique Words.
words = data['token'].unique().tolist() + ['<UNK>', '<PAD>', 'ENDTAG']

# Unique Tags.
tags = data['tag'].unique().tolist()

# Grouped data.
grouped = data.groupby('sentence_no')

# Aggregation function to make a list of tokens in each row.
agg_function_word_only = lambda df: [w for w in df['token'].values.tolist()]

# List of word_lists.
sentences = grouped.apply(agg_function_word_only).tolist()

# Aggregation function to make a list of tags in each row.
agg_function_tag_only = lambda df: [t for t in df['tag'].values.tolist()]
        
# List of tags_tokens.
tag_sequence = grouped.apply(agg_function_tag_only).tolist()

# DataFrame for sentences and tags.
df_sent_tags = pd.DataFrame({'sentences' : sentences, 'tag_sequence' : tag_sequence})

# Remove rows for len less than 3.
id_del = [ii for ii, item in enumerate(sentences) if len(item) < 3]
df_sent_tags.drop(df_sent_tags.index[id_del], inplace=True)