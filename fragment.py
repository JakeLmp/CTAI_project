"""
Annotation data formatting:
-----------------------------------------------------------------------------------------
<interaction>_<idx>.annotations

	+ First line:
		#num_frames: <n>     
	
		<n> :=  number of annotated frames  in the video.

	+ Rest of the file is structured by frames

		#frame: <f>  #num_bbxs: <d> [ #interactions: < id_i - id_j> ]
		<id_1>  <bbx_1>  <label_1>  <ho_1>
		.
		.
		<id_d>  <bbx_d>  <label_d>  <ho_d>

		<f> 			:=  frame number
		<d>			:=  number of upper body annotations in the frame
		<id_i - id_j> :=  id's of people  interacting in this frame (if any)
		<id_i>		:=  person ID corresponding to the i-th bounding box in this frame. 
		<bbx_i>	:=  bounding box dimensions in pixel coordinates: [top_left_x  top_left_y size]
		<label_i> 	:=  interaction label of  i-th annotation		
		<ho_i>		:=  discrete head orientation of i-th annotation 

-----------------------------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd

# import cv2
import matplotlib.pyplot as plt

class Fragment:
    dataset_dir = os.path.normpath(r"C:\Users\Jakob\Downloads\Schoolshait\MSc\Core Topics AI\Project\dataset_filtered")
    ann_dir = os.path.join(dataset_dir, "tv_human_interaction_annotations")
    vid_dir = os.path.join(dataset_dir, "tv_human_interaction_videos")

    fragment_types = ["handShake", "highFive", "hug", "kiss", "negative"]
    
    def __init__(self, file_name):
        # create paths to annotation and video files
        self.ann_file = os.path.join(Fragment.ann_dir, file_name + '.annotations')
        self.vid_file = os.path.join(Fragment.vid_dir, file_name + '.avi')
        
        self.fragment_type = file_name.split('_')[0]
        assert self.fragment_type in Fragment.fragment_types
        self.nr_of_frames = self.get_nr_frames()
        self.nr_of_idx = np.nan # to be set on importing
        
        # create dataframes for annotation data
        self.df_interaction = pd.DataFrame(columns = ['frame', 'num_bbxs', 'interactions', 'involved1', 'involved2'])

        # this later becomes a list of dataframes, one for each idx
        self.df_pose = pd.DataFrame(columns = ['frame', 'idx', 'bbox_x', 'bbox_y', 'bbox_size', 'head_orientation', 'interaction'])
    # ----- ----- ----- -----
    
    def get_nr_frames(self,):
        '''Get the number of frames in the fragment'''
        with open(self.ann_file) as f:
            return int(f.readline().split()[-1])
    # ----- ----- ----- -----
    
    def import_annotations(self,):
        '''
        Import the annotation file, format into dataframes self.df_interactions and self.df_pose.
        
        columns of df_interactions:
        ---------------------------
        'frame' : frame number
        'num_bbxs' : number of bounding boxes
        'interactions' : list of interactions annotated in the frame
        'involved1' : index of participant 1 of the interaction
        'involved2' : index of participant 2 of the interaction
        
        columns of df_pose:
        -------------------
        'frame' : frame number
        'idx' : index of person/bounding box in frame
        'bbox_x' : x-coordinate top-left corner of bounding box
        'bbox_y' : y-coordinate top-left corner of bounding box
        'bbox_size' : size of square bounding box
        'head_orientation' : (profile-left, profile-right, frontal-left, frontal-right, backwards)
        'interaction' : True or False; is there interaction between two people in the frame
        '''
        with open(self.ann_file) as f:
            complete_text = f.readlines()
            
        # delete header
        del complete_text[0]
        
        # going through file line-by-line
        # list of nr of people present in frame
        dlist = []
        for line in complete_text:
            # if frame-header
            if line[0] == '#':
                spline = line.split()
                
                # get frame number and no. of people in frame
                frame = int(spline[1])
                d = int(spline[3])
                dlist.append(d)
                
                # get idx of people interacting, if any
                # also store in a boolean whether there is interaction
                try: 
                    idx1 = int(spline[5])
                    idx2 = int(spline[7])
                    inter_present = True
                except Exception:
                    idx1 = np.nan
                    idx2 = np.nan
                    inter_present = False
                
                # adding to interaction dataframe (columns: ['frame', 'num_bbxs', 'interactions', 'involved1', 'involved2'])
                # 'interactions' is an empty list, values are added when processing the other lines of the frame
                self.df_interaction.loc[len(self.df_interaction)] = [frame, d, [], idx1, idx2]
            
            # if other annotation line
            else:
                spline = line.split()
                
                # get annotations
                idx = int(spline[0])
                bbox = [int(x) for x in spline[1:4]]
                interaction = spline[4]
                head_orientation = spline[5]
                
                # adding to pose dataframe (columns = ['frame', 'idx', 'bbox_x', 'bbox_y', 'bbox_size', 'head_orientation', 'interaction'])
                # 'frame' and 'num_bbxs' should still contain the current frame data
                self.df_pose.loc[len(self.df_pose)] = [frame, idx, bbox[0], bbox[1], bbox[2], head_orientation, inter_present]
                
                # adding notated interaction to interaction dataframe entry
                self.df_interaction.loc[len(self.df_interaction) - 1, 'interactions'].append(interaction)
        
        # convert dtypes
        self.df_interaction = self.df_interaction.convert_dtypes()
        self.df_pose = self.df_pose.convert_dtypes()

        # set frame column as index
        self.df_interaction.set_index('frame', inplace=True)
        self.df_interaction.index = self.df_interaction.index.astype('int64', copy=False)

        # separate by idx, set frame column as index, drop 'idx' column
        self.df_pose = [self.df_pose.loc[self.df_pose['idx'] == i].set_index('frame').drop(labels=['idx'], axis='columns') for i in self.df_pose['idx'].unique()]
        for df in self.df_pose:
            df.index = df.index.astype('int64', copy=False)
        self.nr_of_idx = len(self.df_pose)
    # ----- ----- ----- -----

    def get_interaction_interval(self,):
        """
        Returns starting and ending frames of the interaction in this fragment
        """
        assert not self.df_interaction.empty
        if self.fragment_type == Fragment.fragment_types[-1]: # if fragment is a negative, i.e. contains NO interactions
            return None
        return self.df_interaction.dropna().index.values.tolist()[0], self.df_interaction.dropna().index.values.tolist()[-1]
    # ----- ----- ----- -----

    def get_interaction_participants(self,):
        """
        Returns the idxs of the participants in the interaction in this fragment
        """
        assert not self.df_interaction.empty
        if self.fragment_type == Fragment.fragment_types[-1]: # if fragment is a negative, i.e. contains NO interactions
            return None
        return list(self.df_interaction.dropna().involved1.unique()), list(self.df_interaction.dropna().involved2.unique())
    # ----- ----- ----- -----
    
    def use_only_interaction(self, borders=5):
        """
        Use only the data from during the interaction, with 'borders' frames before and after this interval
        Remove rest of data from dataframes.
        """ 
        
        start, stop = self.get_interaction_interval()
        minFrame = min(self.df_interaction.index.values)
        maxFrame = max(self.df_interaction.index.values)
        
        # check if there are "borders" amount of frames around the interval, otherwise take largest possible border
        start = start - borders
        start = start if start > minFrame else minFrame
        stop = stop + borders
        stop = stop if stop < maxFrame else maxFrame
        
        for df in self.df_pose:
            # to-be-dropped labels are outside of the range [start, stop]
            droplabels = [i for i in df.index.values.tolist() 
                          if (i < start or i > stop)]
            
            df.drop(labels=droplabels, axis='index', inplace=True)
    # ----- ----- ----- -----
    
    def plot_trajectory(self, idx=None, showfig=True, savefig=False, savename=None, savedir=None):
        """
        Scatterplot trajectories of people in frame
        """ 
        if not idx:
            idx = []
        
        # check if dataframes not empty
        if self.df_interaction.empty or len(self.df_pose)==0 or any([df.empty for df in self.df_pose]):
            raise Exception('annotations not yet imported')
        
        fig, ax = plt.subplots()
        
        plot_idx = range(len(self.df_pose)) if not idx else idx
        for i in plot_idx:
            ax.scatter(x=self.df_pose[i]['bbox_x'], y=self.df_pose[i]['bbox_y'], label=str(i))
        
        ax.axis('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # ax.legend(title='idx', bbox_to_anchor=(1,1), loc="upper left")
        
        if showfig: plt.show()
        if savefig: 
            assert savename is not None
            assert os.path.isdir(savedir)
            savedir = os.path.normpath(savedir)
            plt.savefig(os.path.join(savedir, savename))
        
        plt.close(fig)
    # ----- ----- ----- -----

    def plot_signals(self, idx=None, interaction_borders=True, showfig=True, savefig=False, savename=None, savedir=None):
        """
        Line plots of separate signals

        'signals' : what columns of df_pose to plot
        """
        if not idx:
            idx = []
        
        # check if dataframes not empty
        if self.df_interaction.empty or len(self.df_pose)==0 or any([df.empty for df in self.df_pose]):
            raise Exception('annotations not yet imported')
        
        fig, ax = plt.subplots(nrows=2, sharex=True)
        
        # add vertical lines at start and end of interaction
        if interaction_borders: 
            interval = self.get_interaction_interval()
            ax[0].axvline(x=interval[0], linestyle='dotted')
            ax[0].axvline(x=interval[1], linestyle='dotted')
            ax[1].axvline(x=interval[0], linestyle='dotted')
            ax[1].axvline(x=interval[1], linestyle='dotted')
        
        plot_idx = range(len(self.df_pose)) if not idx else idx
        # print(plot_idx)
        for i in plot_idx:
            # for some reason, matplotlib wants the pd.Series converted to lists. I'm not asking why.
            ax[0].plot(self.df_pose[i].index.values.tolist(), list(self.df_pose[i]['bbox_x']), label=str(i))
            ax[1].plot(self.df_pose[i].index.values.tolist(), list(self.df_pose[i]['bbox_y']), label=str(i))

        # ax[0].set_title()
        ax[1].set_xlabel('frame')
        ax[0].set_ylabel('x')
        ax[1].set_ylabel('y')
        # ax[0].legend(title='idx', bbox_to_anchor=(1,1), loc="upper left")

        if showfig: plt.show()
        if savefig: 
            assert savename is not None
            assert os.path.isdir(savedir)
            savedir = os.path.normpath(savedir)
            plt.savefig(os.path.join(savedir, savename))
        
        plt.close(fig)
    # ----- ----- ----- -----
    
    def dist(cls, x1: float, x2: float, y1: float, y2: float, round_to=3) -> float:
        """Euclidean distance in 2 dimensions. Rounded to 3 decimals by default."""
        return round(np.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)), 3)
    # ----- ----- ----- -----
    
    def _diff(self, idx):
        """
        Take Euclidean distance between data points i and i-1 of bounding box trajectory idx.
        Returns Series of length N-1 (if trajectory is length N), starting at the second frame in the original Series
        """
        # check if dataframes not empty
        if self.df_interaction.empty or len(self.df_pose)==0 or any([df.empty for df in self.df_pose]):
            raise Exception('annotations not yet imported')
        
        # breakpoint()
        # temp = self.df_pose[idx].bbox_y
        
        new = [self.dist(self.df_pose[idx].bbox_x[i-1],
                         self.df_pose[idx].bbox_x[i],
                         self.df_pose[idx].bbox_y[i-1], 
                         self.df_pose[idx].bbox_y[i]) for i in self.df_pose[idx].index.values.tolist()[1:]]
        
        new_index = pd.Index(self.df_pose[idx].index.values.tolist()[1:], name='frame')
        return pd.Series(new, index=new_index)
    # ----- ----- ----- -----
    
    def add_diff_cols(self,):
        """
        Add new columns to df_pose dataframes containing the Euclidean distances as defined in self._diff()
        """
        # check if dataframes not empty
        if self.df_interaction.empty or len(self.df_pose)==0 or any([df.empty for df in self.df_pose]):
            raise Exception('annotations not yet imported')
        
        for i, df in enumerate(self.df_pose):
            df['diff'] = self._diff(i)
    # ----- ----- ----- -----
    
    def plot_diff(self, idx=None, interaction_borders=True, showfig=True, savefig=False, savename=None, savedir=None):
        """
        Plot the differencing data
        """
        if not idx:
            idx = []
        
        # check if dataframes not empty
        if self.df_interaction.empty or len(self.df_pose)==0 or any([df.empty for df in self.df_pose]):
            raise Exception('annotations not yet imported')
        
        if not all(('diff' in df for df in self.df_pose)):
            self.add_diff_cols()
        
        fig, ax = plt.subplots()
        
        # add vertical lines at start and end of interaction
        if interaction_borders: 
            interval = self.get_interaction_interval()
            ax.axvline(x=interval[0], linestyle='dotted')
            ax.axvline(x=interval[1], linestyle='dotted')
        
        plot_idx = range(len(self.df_pose)) if not idx else idx
        for i in plot_idx:
            ax.plot(self.df_pose[i].index.values.tolist(), list(self.df_pose[i]['diff']), label=str(i))
        
        # ax.axis('equal')
        ax.set_xlabel('frame')
        ax.set_ylabel(r'$\Delta$')
        
        if showfig: plt.show()
        if savefig: 
            assert savename is not None
            assert os.path.isdir(savedir)
            savedir = os.path.normpath(savedir)
            plt.savefig(os.path.join(savedir, savename))
        
        plt.close(fig)
    # ----- ----- ----- -----
# ----- ----- ----- -----