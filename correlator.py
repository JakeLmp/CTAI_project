import numpy as np
from numpy.fft import fft, ifft

import pandas as pd
from itertools import product
from fragment import Fragment

class Correlator:
    """Correlation functions for the Fragment class"""
    def __init__(self, fragment):
        # assert isinstance(fragment, Fragment)
        
        self.fragment = fragment

        # check if dataframes not empty (assuming that if this one is not empty, the rest isn't either)
        if self.fragment.df_interaction.empty:
            self.fragment.import_annotations()
        
        # attribute to store the most recently executed correlation results
        self.df = None

    def _correlate_lagged_series(self, s1, s2, min_periods=None):
        """
        Correlate given series after applying time shifts. 
        If min_periods not given, do this for all sensible time-lags.
        Else, do for all time-lags allowed by min_periods
        """
        assert isinstance(s1, pd.Series)
        assert isinstance(s2, pd.Series)
        
        s1 = s1.astype(float)
        s2 = s2.astype(float)
        
        # largest time-shift we can do is the length of the shortest series, minus 1
        max_lag = len(s1)-1 if len(s1) < len(s2) else len(s2)-1
        
        # create output series
        sout = pd.Series(index=list(range(-max_lag+1, max_lag)), dtype='float64')
        
        # for all negative time-shifts
        for lag in range(-max_lag+1, 0):
            s2_temp = s2.copy()
            s2_temp.index = [x + lag for x in s2.index.values.tolist()]
            sout[lag] = s1.corr(s2_temp, min_periods=min_periods)
        
        sout[0] = s1.corr(s2, min_periods=min_periods)

        # for all positive time-shifts
        for lag in range(1, max_lag):
            s2_temp = s2.copy()
            s2_temp.index = [x + lag for x in s2.index.values.tolist()]
            sout[lag] = s1.corr(s2_temp, min_periods=min_periods)

        return sout
    # ----- ----- ----- -----

    def _correlate_all_series(self, df1, df2, ignore=[]):
        """
        Correlate the columns of df1 with the columns of df2, 
        after applying time shifts to the columns of df2.
        
        Ignore the columns given in 'ignore'.
        """
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)

        # list of columns that we are not ignoring
        cols = [col for col in df1.columns if col not in ignore]
        
        # make list of possible combinations (including combining duplicates)
        colvcol = list(product(cols, cols))
        
        # make list of output column names
        colvcol_comb = [x[0] + '_1 - ' + x[1] + '_2' for x in colvcol]
        
        # create output dataframe
        df = pd.DataFrame(columns=colvcol_comb)
        df.index.name = 'frame shift'
        
        # do the time shifting and correlating for all combinations
        for pair, col in zip(colvcol, colvcol_comb):
            df[col] = self._correlate_lagged_series(df1[pair[0]], df2[pair[1]])
        
        return df
    # ----- ----- ----- -----
    
    def fft_whitening(self, data):
        """
        Perform fft whitening on given data.
        Expects pd.Series, casts whitened data into copy of original objet.
        """
        assert isinstance(data, pd.Series)
        
        f = fft(data)   # calc frequency domain
        f1 = f / np.abs(f)  # normalise in frequency domain
        res = data.copy()
        res = np.real(ifft(f1))     # calc whitened data from normalised frequency domain, return
        return res
    # ----- ----- ----- -----
    
    def correlate_columns(self, col1, col2):
        """Execute the lagged cross correlation on column 'col1' of interaction participant 1 and 'col2' of interaction participant 2"""
        part1, part2 = self.fragment.get_interaction_participants()
        
        self.df = self._correlate_lagged_series(self.fragment.df_pose[part1[0]][col1], self.fragment.df_pose[part2[0]][col2])
        
        return self.df
    
    def correlate_bounding_boxes(self, idx1, idx2):
        """Execute the lagged cross correlation on the bounding box movements of persons with idx1 and idx2"""
        ignore = ['bbox_size', 'head_orientation', 'interaction']

        self.df = self._correlate_all_series(self.fragment.df_pose[idx1], self.fragment.df_pose[idx2], ignore=ignore)

        return self.df
    # ----- ----- ----- -----
    
    def correlate_diff(self, idx1=None, idx2=None):
        """
        Execute the lagged cross correlation on the bounding box frame differences, using data from idx1 and idx2.
        If idx1 and/or idx2 are not given, the interaction participants are used.
        """
        if not idx1 and not idx2:
            idx1, idx2 = self.fragment.get_interaction_participants()
            idx1 = idx1[0]
            idx2 = idx2[0]
        
        if not all(['diff' in df for df in self.fragment.df_pose]):
            self.fragment.add_diff_cols()
        
        self.df = self._correlate_lagged_series(self.fragment.df_pose[idx1]['diff'], 
                                                self.fragment.df_pose[idx2]['diff'])
        
        return self.df
    # ----- ----- ----- -----
    