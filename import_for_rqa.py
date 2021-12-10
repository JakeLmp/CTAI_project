# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:18:31 2021

@author: J.H.B. Limpens (jakob.limpens@gmail.com)
"""

import pandas as pd
import numpy as np

import os
from tqdm import tqdm

import matplotlib.pyplot as plt

#%% IMPORTING DATA

savedir = r"C:\Users\Jakob\Downloads\Schoolshait\MSc\Core Topics AI\Project\recurrence_stuff"
file = os.path.join(savedir, 'cross_rec.csv')

df = pd.read_csv(file, index_col=0)

# splitting into list of dataframes, separated by interaction type
df_split = []
fragtypes = ['handShake', 'highFive', 'hug', 'kiss']
for t in fragtypes:
    temp = []
    for f in df.index.values.tolist():
        if f.startswith(t):
            temp.append(f)
    
    # drop nans and append a copy of the selection to list of dataframes
    df_split.append(df.loc[temp, :].dropna().copy())

#%% PLOTTING BARS

tot_mean = df.mean(axis=0)
tot_errs = df.std(axis=0)

means = [D.mean(axis=0) for D in df_split]
errs = [D.std(axis=0) for D in df_split]

# ['recurrence_rate', 'determinism', 'divergence', 'average_len', 'longest_len', 'laminarity']
current = 'recurrence_rate'
x = [t[current] for t in means]
err = [t[current] for t in errs]

fragtypes = ['handShake', 'highFive', 'hug', 'kiss']

plt.grid(visible=True, which='major', axis='y', alpha=0.5)
plt.tick_params(direction='in')
plt.axhline(tot_mean[current], color='black', linestyle='dotted')
plt.bar(x=fragtypes, height=x, yerr=err, error_kw={'barsabove':True})


tits = {'recurrence_rate':'recurrence rate', 
        'determinism':'determinism', 
        'divergence':'divergence',
        'average_len':'average length',
        'longest_len':'longest length',
        'laminarity':'laminarity'}
# plt.title(f'Mean {tits[current]} of interaction types')


