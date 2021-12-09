# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 11:16:34 2021

@author: J.H.B. Limpens (jakob.limpens@gmail.com)
"""

import pandas as pd
import numpy as np

import os
from tqdm import tqdm

import matplotlib.pyplot as plt

from fragment import Fragment
from correlator import Correlator

# iterator that gets the filenames without the extensions
fnames = sorted((os.path.splitext(os.path.basename((f)))[0] for f in os.scandir(Fragment.ann_dir)))

df = pd.DataFrame()

minimum_length = 20

for file in tqdm(fnames):
    frag = Fragment(file)
    frag.import_annotations()
    frag.add_diff_cols()
    
    # checking if the interaction itself is long enough, increasing borders if it isn't
    start, stop = frag.get_interaction_interval()
    border = 5
    if stop-start <= minimum_length:
        border += (minimum_length - (stop-start)) / 2
    
    frag.use_only_interaction(borders=border)
    
    part1, part2 = frag.get_interaction_participants()
    new_sr1 = frag.df_pose[part1[0]]['diff']
    new_sr1.name = file + '_1'
    new_sr2 = frag.df_pose[part2[0]]['diff']
    new_sr2.name = file + '_2'
    df = pd.concat([df, new_sr1, new_sr2], axis='columns')
    
savedir = os.getcwd()
df.to_csv(os.path.join(savedir, 'interaction_diffs.csv'))