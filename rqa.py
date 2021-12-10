#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 11:13:00 2021

@author: ronja
"""

import pandas as pd

from pyrqa.computation import RQAComputation, RPComputation
from pyrqa.image_generator import ImageGenerator
from pyrqa.analysis_type import Cross
from pyrqa.neighbourhood import FixedRadius
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.metric import EuclideanMetric

path = "/Users/ronja/Desktop/Art-Int/jakobs code/interaction_diffs.csv"

interactions = pd.read_csv(path, index_col=0)
#interactions = interactions.iloc[:,:10]

em = 2
d = 2
r = 1

types = ["handShake","highFive","hug","kiss"]

output_df = pd.DataFrame(columns=["recurrence_rate","determinism","divergence","average_len","longest_len","laminarity"])

for i in range(0, interactions.shape[1], 2):
    df = interactions.iloc[:,i:i+2]
    df.dropna(inplace=True)
    frag_name = df.columns[0][:-2]
    
    time_series_x = TimeSeries(df.iloc[:,0],
                               embedding_dimension=em,
                               time_delay=d)
    time_series_y = TimeSeries(df.iloc[:,1],
                               embedding_dimension=em,
                               time_delay=d)
    time_series = (time_series_x,
                   time_series_y)

    settings = Settings(time_series,
                        analysis_type=Cross,
                        neighbourhood=FixedRadius(r),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=0)

    computation = RQAComputation.create(settings,verbose=False)
    result = computation.run()
    result.min_diagonal_line_length = 2
    result.min_vertical_line_length = 2
    result.min_white_vertical_line_length = 2
    
    ap_df = pd.DataFrame({"recurrence_rate":result.recurrence_rate,
                          "determinism":result.determinism,
                          "divergence":result.divergence,
                          "average_len":result.average_diagonal_line,
                          "longest_len":result.longest_diagonal_line,
                          "laminarity":result.laminarity}, index=[frag_name])
    
    output_df = output_df.append(ap_df)
    
    computation = RPComputation.create(settings)
    result = computation.run()
    ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse,
                                        '/Users/ronja/Desktop/Art-Int/jakobs code/recurrence_plot_{}.png'.format(frag_name))


output_df.to_csv("/Users/ronja/Desktop/Art-Int/jakobs code/cross_rec.csv")
