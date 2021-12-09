# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 12:52:57 2021

@author: J.H.B. Limpens (jakob.limpens@gmail.com)
"""

import pandas as pd
import numpy as np

import os
from tqdm import tqdm

import matplotlib.pyplot as plt

from fragment import Fragment
from correlator import Correlator

#%% RUNNING CORRELATION ON DIFFs

# iterator that gets the filenames without the extensions
fnames = sorted((os.path.splitext(os.path.basename((f)))[0] for f in os.scandir(Fragment.ann_dir)))

# splitting by interaction type
split_fnames = []
for typ in Fragment.fragment_types:
    temp = []
    with os.scandir(Fragment.ann_dir) as it:
        for entry in it:
            # if current entry starts with current fragment type, and is a file
            if entry.name.startswith(typ) and entry.is_file():
                temp.append(os.path.splitext(os.path.basename((entry)))[0])
    
    # add to the main list
    split_fnames.append(temp)

df_list = []
minimum_length = 20

for file_list in split_fnames:
    # new dataframe 
    df = pd.DataFrame()
    
    # do the correlations for all the fragments in the current list
    with np.errstate(divide='ignore', invalid='ignore'):
        for f in tqdm(file_list):
            frag = Fragment(f)
            frag.import_annotations()
            frag.add_diff_cols()
            
            # checking if the interaction itself is long enough, increasing borders if it isn't
            start, stop = frag.get_interaction_interval()
            border = 5
            if stop-start <= minimum_length:
                border += (minimum_length - (stop-start)) / 2
            
            frag.use_only_interaction(borders=border)
            
            corr = Correlator(frag)
            new_sr = corr.correlate_diff()
            df = pd.concat([df, new_sr], axis='columns')
    
    df.columns = file_list
    
    df_list.append(df)

#%% RUNNING CORRELATION ON Xs

# iterator that gets the filenames without the extensions
fnames = sorted((os.path.splitext(os.path.basename((f)))[0] for f in os.scandir(Fragment.ann_dir)))

# splitting by interaction type
split_fnames = []
for typ in Fragment.fragment_types:
    temp = []
    with os.scandir(Fragment.ann_dir) as it:
        for entry in it:
            # if current entry starts with current fragment type, and is a file
            if entry.name.startswith(typ) and entry.is_file():
                temp.append(os.path.splitext(os.path.basename((entry)))[0])
    
    # add to the main list
    split_fnames.append(temp)

df_list = []
minimum_length = 20

for file_list in split_fnames:
    # new dataframe 
    df = pd.DataFrame()
    
    # do the correlations for all the fragments in the current list
    with np.errstate(divide='ignore', invalid='ignore'):
        for f in tqdm(file_list):
            frag = Fragment(f)
            frag.import_annotations()
            
            # checking if the interaction itself is long enough, increasing borders if it isn't
            start, stop = frag.get_interaction_interval()
            border = 5
            if stop-start <= minimum_length:
                border += (minimum_length - (stop-start)) / 2
            
            frag.use_only_interaction(borders=border)
            
            corr = Correlator(frag)
            new_sr = corr.correlate_columns('bbox_x', 'bbox_x')
            df = pd.concat([df, new_sr], axis='columns')
    
    df.columns = file_list
    
    df_list.append(df)


#%% SAVE TO CSVs

savedir = os.path.normpath(r"C:\Users\Jakob\Downloads\Schoolshait\MSc\Core Topics AI\Project\data_processed")

for df, name in zip(df_list, Fragment.fragment_types):
    filename = os.path.join(savedir, f'{name}.csv')
    df.to_csv(filename)

#%% RUNNING CORRELATION ON NEGATIVES

# iterator that gets the filenames without the extensions
dirname = os.path.join(os.getcwd(), 'dataset', 'tv_human_interaction_annotations')
fnames = sorted((os.path.splitext(os.path.basename((f)))[0] for f in os.scandir(dirname) if os.path.basename((f)).startswith('negative')))

df = pd.DataFrame()
minimum_length = 20

with np.errstate(divide='ignore', invalid='ignore'):
    for file in tqdm(fnames):
        frag = Fragment(file)
        frag.import_annotations()
        
        # skip this fragment if less than 2 people in it, or if shorter than 20 frames
        if frag.nr_of_idx < 2 or frag.nr_of_frames < 20:
            continue
            
        frag.add_diff_cols()
        
        corr = Correlator(frag)
        new_sr = corr.correlate_diff(idx1=0, idx2=1)
        new_sr.name = file
        
        # skip this fragment if resulting correlation Series is shorter than minimum length
        if len(new_sr) < minimum_length:
            continue
        
        df = pd.concat([df, new_sr], axis='columns')

# save to csv
filename = os.path.join(os.getcwd(), 'data_processed', 'diff', 'negative.csv')
df.to_csv(filename)

#%% READ FROM CSVs

savedir = os.path.normpath(r"C:\Users\Jakob\Downloads\Schoolshait\MSc\Core Topics AI\Project\data_processed\diff")

# file iterator
it = os.scandir(savedir)

# select which CSV to read: ['handShake', 'highFive', 'hug', 'kiss', 'negative']
select = 'negative'

for entry in it:
    if entry.name.startswith(select) and entry.is_file():
        # print(entry.path)
        df = pd.read_csv(entry.path, index_col=0)

#%% PREPROCESSING DATAFRAME

# for now, we're dropping NA-containing rows
# this means we're only looking at time lags as long as the shortest interaction can handle
# df.dropna(axis='index', inplace=True)

# alternatively, drop everything outside of the [-15,15] range
start = -15     # inclusive
stop = 15       # inclusive
droplabels = [i for i in df.index.values.tolist()
              if i < start or i > stop]
df.drop(index=droplabels, inplace=True)


#%% PLOTTING

ptr = df

# AGGREGATED PLOT
# ci = 1.96/np.sqrt(df.shape[1])
# plt.axhline(y=ci, linestyle='dotted')
# plt.axhline(y=-ci, linestyle='dotted')

x = ptr.index.values.tolist()
y = ptr.mean(axis=1)
error = ptr.std(axis=1)

# getting location + value of maximum of mean
ymax = max(y)
xmax = y.idxmax(ymax)

# plotting lines at maximum
plt.axvline(x=xmax, color='orange', linestyle='dotted')
plt.axhline(y=ymax, color='orange', linestyle='dotted')

plt.plot(x, y)

plt.xlim([x[0], x[-1]])
plt.ylim([-1, 1])

# plt.xticks(np.arange(-0.5, 0.51, step=0.25))
plt.yticks(np.arange(-1, 1.01, step=0.5))
plt.grid(visible=True, which='both')
plt.tick_params(direction='in')

plt.xlabel('Time lag (in frames)')
plt.ylabel('Correlation coefficient')
plt.title(f'Interaction type: {select}, maximum {round(ymax,2)} at frame lag {xmax}')

plt.fill_between(x, y-error, y+error, alpha=0.25)

plt.show()

