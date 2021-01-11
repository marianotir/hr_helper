# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 12:19:07 2021

@author: Mariano
"""

#------------------------
# Load Initial Libraries
#------------------------

import pandas as pd 
import numpy as np
from sklearn.compose import make_column_transformer


#----------------------
# Load data
#----------------------

df = pd.read_csv('C:/Users/Mariano/DS_Models/St_HR/HRDataset.csv')  


#-------------------------
# Preprocess Data
#-------------------------

# Delete unnecesary variables
df = df.drop(columns=['Unnamed: 0'])

# Check apareance
df.head(10)

# description
description = df.describe()
print(description)

# Check column types
df.dtypes

# Check shape
df.shape


#-----------------------------
# Feature engineering
#-----------------------------

# Encode categorical variable 

    # Get the list of categorical descriptive features
categorical_cols = df.columns[df.dtypes==object].tolist()

df.PerformanceScore.unique()

    # Dictionary map for the label encoding variable
map_dict = {'Exceeds': 4, 'Fully Meets': 3,'Needs Improvement': 2,'PIP': 1}

    # Use dictionary map_dict to map Object data type to int8
df['PerformanceScore'] = df['PerformanceScore'].map(map_dict)


#-----------------------------
# Prepare data for training
#-----------------------------

# Target and feats variables
from sklearn import preprocessing

X = df.iloc[:,0:-1].values
Y = df.iloc[:,-1].values

le = preprocessing.LabelEncoder()
le_fit = le.fit(Y)
target_encoded_le = le_fit.transform(Y)


# Split for training 
from sklearn.model_selection import train_test_split

X_train,X_test, Y_train, Y_test = train_test_split(X,target_encoded_le,test_size=0.33)


#----------------------
# Train model
#----------------------
from sklearn.ensemble import RandomForestClassifier

# Model results before umbalance

model_rf = RandomForestClassifier()
model_rf.fit(X_train,Y_train)

pred_train = model_rf.predict(X_train)
pred       = model_rf.predict(X_test)

# Modeol results after unbalance

  # Checking accuracy
from sklearn.metrics import accuracy_score
print('accuracy_score =',accuracy_score(Y_test, pred))

  # f1 score
from sklearn.metrics import f1_score
print('f1_score =',f1_score(Y_test, pred))

  # recall score
from sklearn.metrics import recall_score
print('recall_score =',recall_score(Y_test, pred))

# Solve umbalance 
# Applky Synthetic Minority Oversampling Technique
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=27)
X_train, Y_train = sm.fit_sample(X_train, Y_train)


# Model results after unbalance

model_rf = RandomForestClassifier(n_estimators = 50, random_state = 0)
model_rf.fit(X_train,Y_train)

pred_train = model_rf.predict(X_train)
pred       = model_rf.predict(X_test)

  # Checking accuracy
from sklearn.metrics import accuracy_score
print('accuracy_score =',accuracy_score(Y_test, pred))

  # f1 score
from sklearn.metrics import f1_score
print('f1_score =',f1_score(Y_test, pred))

  # recall score
from sklearn.metrics import recall_score
print('recall_score =',recall_score(Y_test, pred))

# Note: Since random forest model is applied there is no need for normalization or scale the data 
#       before training

#-----------------------w
# Save model
#-----------------------
import pickle
pkl_filename = "C:/Users/Mariano/DS_Models/St_HR/model_rf_10012020.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model_rf, file)
    
import joblib
joblib_file = "C:/Users/Mariano/DS_Models/St_HR/model_rf_10012020.pkl"  
joblib.dump(model_rf, joblib_file)

#--------------------------
# Load the model
#--------------------------

model = joblib.load(pkl_filename)

# Generate predictions 
PerformanceScore = 4
EmpSatisfaction  = 4
EngagementSurvey = 4

data = {'PerformanceScore': PerformanceScore,
        'EmpSatisfaction': EmpSatisfaction,
        'EngagementSurvey': EngagementSurvey}
    
# Data for prediction output
df = pd.DataFrame(data,index=[0])
X_outsample = df.values
    
pred = model.predict(X_outsample)
pred





