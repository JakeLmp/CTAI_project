# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 11:47:53 2021

@author: J.H.B. Limpens (jakob.limpens@gmail.com)
"""

import os
# set current working directory
os.chdir(r"C:\Users\Jakob\Downloads\Schoolshait\MSc\Core Topics AI\Project")

import pandas as pd
import numpy as np
# import pathlib

from fragment import Fragment
from correlator import Correlator

from tqdm import tqdm

#%% ----- FILTERING OUT FRAGMENTS CONTAINING CUTS -----
import shutil

# define and create new directories
new_dir = os.path.join(os.getcwd(), 'dataset_filtered')
if not os.path.isdir(new_dir):
    os.mkdir(new_dir)

new_ann_dir = os.path.join(new_dir, "tv_human_interaction_annotations")
if not os.path.isdir(new_ann_dir):
    os.mkdir(new_ann_dir)
    
new_vid_dir = os.path.join(new_dir, "tv_human_interaction_videos")
if not os.path.isdir(new_vid_dir):
    os.mkdir(new_vid_dir)
    
old_dir = os.path.join(os.getcwd(), 'dataset')
old_ann_dir = os.path.join(old_dir, "tv_human_interaction_annotations")
old_vid_dir = os.path.join(old_dir, "tv_human_interaction_videos")

# iterator that gets the filenames without the extensions
fnames = (os.path.splitext(os.path.basename((f)))[0] for f in os.scandir(old_ann_dir))

# iterators that get the full file paths of the original files
ann_fullnames = (f for f in os.scandir(old_ann_dir))
vid_fullnames = (f for f in os.scandir(old_vid_dir))

for f, ann, vid in tqdm(zip(fnames, ann_fullnames, vid_fullnames)):
    frag = Fragment(f)
    frag.import_annotations()
    
    part = frag.get_interaction_participants()
    # if negative fragment, skip copying
    if not part:
        continue
    # if fragment contains cuts, skip copying
    if len(part[0]) > 1 or len(part[1]) > 1:
        continue
    
    # else, copy files into new directory
    shutil.copy2(ann, new_ann_dir)
    shutil.copy2(vid, new_vid_dir)


#%% ----- GENERATING RAW DATA PLOTS -----

# iterator that gets the filenames without the extensions
fnames = (os.path.splitext(os.path.basename((f)))[0] for f in os.scandir(Fragment.ann_dir)) 

# upper directory in which to save images
img_dir = os.path.join(os.getcwd(), 'images')

for f in tqdm(sorted(fnames)[:]):
    fragment = Fragment(f)
    fragment.import_annotations()
    
    part = fragment.get_interaction_participants()
    part = part[0] + part[1] if part is not None else None
    
    # showfig=True, savefig=False, savename=None, savedir=None
    
    # plot full signals
    savedir = os.path.join(img_dir, 'full_signals')
    fragment.plot_signals(idx=[], showfig=False, savefig=True, savename='full_signal_' + f + '.png', savedir=savedir)

    # plot interaction signals
    savedir = os.path.join(img_dir, 'interaction_signals')
    fragment.plot_signals(idx=part, showfig=False, savefig=True, savename='full_signal_' + f + '.png', savedir=savedir)

    # plot full trajectories
    savedir = os.path.join(img_dir, 'full_trajectories')
    fragment.plot_trajectory(idx=[], showfig=False, savefig=True, savename='full_trajectory_' + f + '.png', savedir=savedir)

    # plot interaction trajectories
    savedir = os.path.join(img_dir, 'interaction_trajectories')
    fragment.plot_trajectory(idx=part, showfig=False, savefig=True, savename='full_trajectory_' + f + '.png', savedir=savedir)


#%% ----- DOING CROSS-CORRELATION STUFF -----

# iterator that gets the filenames without the extensions
fnames = (os.path.splitext(os.path.basename((f)))[0] for f in os.scandir(Fragment.ann_dir))

for f in tqdm(sorted(fnames)[:]):
    fragment = Fragment(f)
    fragment.import_annotations()
    
    part = fragment.get_interaction_participants()
    # part = part[0] + part[1] if part is not None else None
    
    correlator = Correlator(fragment)
    correlator.correlate_bounding_boxes(part)
    
#%% ----- GENERATING CSVs (FOR CRQA IN R) -----

# iterator that gets the filenames without the extensions
fnames = (os.path.splitext(os.path.basename((f)))[0] for f in os.scandir(Fragment.ann_dir))

# directory to write CSVs to
csv_dir = os.path.join(os.getcwd(), 'dataset_filtered', 'CSVs')
if not os.path.isdir(csv_dir):
    os.mkdir(csv_dir)

for f in tqdm(sorted(fnames)[:]):
    fragment = Fragment(f)
    fragment.import_annotations()
    fragment.add_diff_cols()
    
    # we're going to store all CSVs of the current fragment in a separate subfolder
    curr_dir = os.path.join(csv_dir, f)
    if not os.path.isdir(curr_dir):
        os.mkdir(curr_dir)
    
    # for each person in the fragment, write a new CSV named 'pose_{index}'
    for i, df in enumerate(fragment.df_pose):
        file_name = os.path.join(curr_dir, f'pose_{i}.csv')
        df.to_csv(file_name, index_label=False) # no index name for easier importing in R
    
    # we're also exporting the interaction dataframe, because why not
    file_name = os.path.join(curr_dir, 'interaction.csv')
    fragment.df_interaction.to_csv(file_name, index_label=False)
    
    
    