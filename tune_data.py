# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:09:27 2021

@author: Mariano
"""

import pandas as pd 
import numpy as np


df = pd.read_csv('C:/Users/Mariano/DS_Models/St_HR/HRDataset_v14.csv')

df.dtypes

df.EngagementSurvey
df.EmpSatisfaction


  
df1 = df[[
          'PerformanceScore',
          'EmpSatisfaction',
          'EngagementSurvey',
          'Termd'
          ]]


df1.to_csv('C:/Users/Mariano/DS_Models/St_HR/HRDataset.csv')  