# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:34:48 2021

@author: Mariano
"""


from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd


def imput_dataset(df):

    df_num = df.select_dtypes(include=np.number)
    
    df_cat = df.select_dtypes(include=['object'])
    
    
    # clean dataframe using imputer 
    
    #numerical imputer ::: simple way df_mean_imputed = df.fillna(df.mean())
    imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    imp_fit = imp_num.fit_transform(df_num)
    
    df_numerical = pd.DataFrame(imp_fit,columns=df_num.columns)
    
    # categorical imputer
    imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    
    imp_fit = imp_cat.fit_transform(df_cat)
    
    df_categorical = pd.DataFrame(imp_fit,columns=df_cat.columns)
    
    
    df_clean = pd.concat([df_numerical,df_categorical], axis=1)
    
   
    return df_clean