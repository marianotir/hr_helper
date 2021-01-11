# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:40:40 2021

@author: Mariano
"""

#-------------------------------
# Import libraries
#-------------------------------

# Deploy model libraries
import streamlit as st
import joblib

# Model libraries
from sklearn.ensemble import RandomForestClassifier

# Dataframe manipulation libraries
import pandas as pd

#-------------------------------
# StreamLit Application
#-------------------------------

# Load model 
model = joblib.load('model_rf_10012020.pkl')

def main():

    # Aplication header
    st.write("""
             # Employee Leaving Prediction App 
             """)
             
    # Sidebar parameters
    st.sidebar.header('User input parameters')
    
    # Parameters
    PerformanceScore = st.sidebar.slider('Performance', 1, 2, 4)
    EmpSatisfaction = st.sidebar.slider('Satisfaction', 1, 2, 5)
    EngagementSurvey = st.sidebar.slider('Enganment', 1, 2, 5)
    data = {'PerformanceScore': PerformanceScore,
            'EmpSatisfaction': EmpSatisfaction,
            'EngagementSurvey': EngagementSurvey}
    
    # Data for prediction output
    df = pd.DataFrame(data,index=[0])
    
    st.subheader(' Data input parameters')
    st.write(df)
    
    
    # Generate predictions on unseen data
    X_outsample = df.values
    
    predictions = model.predict(X_outsample)
    
    prediction_output = predictions[0]

    if prediction_output > 0:
        prediction_output = 'Leaving probable'
    else: 
        prediction_output = 'Staying probable'
    st.subheader(' Prediction Output')
    st.write(prediction_output)
    
    
if __name__=='__main__':
    main()












