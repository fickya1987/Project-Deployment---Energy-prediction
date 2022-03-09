# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:48:46 2022

@author: 91890
"""

import streamlit as st
import pandas as pd


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px


header   = st.container()
dataset  = st.container()
features = st.container()
Model    = st.container()




st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 90%;
        padding-top: 5rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 5rem;
    }}
    img{{
    	max-width:40%;
    	margin-bottom:40px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )

@st.cache(suppress_st_warning=True)
def get_data(filename):
    energy_data = pd.read_csv(filename,sep=';') 
    return energy_data
    
    
with header:
    st.title('Combined Cycle Power Plant Energy Prediction')
    

with dataset:
    st.header('Energy Dataset')
    
    energy_data = get_data('energy_production.csv',    )
    st.write(energy_data.head(50))
    st.write(energy_data.shape)
    
    # EDA
    st.header('EDA')
    st.subheader('Histogram Distributions')

    #Histogram Visualization
    sel_col,dis_col = st.columns(2)
    colums = energy_data.columns
    hist_col = sel_col.selectbox('Select The Column for Visulization', options= colums,  index=0 )
    
    hist_plots = px.histogram(energy_data,x = hist_col, )
    dis_col.write(hist_plots)
    
    # Boxplot Visulization
    st.subheader('Boxplots to check for outliers')
    sel_col1,dis_col1 = st.columns(2)
    box_col = sel_col1.selectbox('Select The Column for Visulization', options= colums, key = 'start_box' ,  index=0 )
    
    box_plots = px.box(energy_data,y = box_col, )
    dis_col1.write(box_plots)
    
    # Scatter plots
    st.subheader("Scatter Plots")
    sel_col2,dis_col2 = st.columns(2)
    scat_col = sel_col2.selectbox('Select The Column for Visulization', options= colums, key = 'start_scat' ,  index=0 )
    
    scat_plots = px.scatter(energy_data, x= 'energy_production' ,y = scat_col, trendline='ols' )
    dis_col2.write(scat_plots)      
    
    # Heatmap
    st.subheader("Heatmap")
    corr_matrix = energy_data.corr()
    heatmap = px.imshow(corr_matrix, text_auto= True)
    st.write(heatmap)    

    # Model Building

with Model:
    sel_col3,dis_col3 = st.columns(2)
    
    st.header("Random Forest Model")
    
    X = energy_data.iloc[:,:-1]
    Y = energy_data.loc[:,'energy_production']

   
    
    # Slider for Max_depth
    
    max_depth = sel_col3.slider('Select The Max_depth', min_value= 2,  max_value= 100 ,value=10, step= 2 )
    n_estimators = sel_col3.selectbox('Select the Number of Trees', options= [100,150,200,250,300,350,400,450,500], index = 0)
    #criterion =  sel_col3.selectbox('Select the Metrics', options = ['mae','mse'], index= 0 )
    
    rf_final_model = RandomForestRegressor(n_estimators=n_estimators,criterion= 'mae',max_depth=max_depth,random_state=21)
    rf_final_model.fit(X,Y)
    
    y_pred_rf_final = rf_final_model.predict(X)
    
    dis_col3.subheader('The R2 Score for the Model is')
    dis_col3.write(r2_score(Y,y_pred_rf_final))
    
    dis_col3.subheader('The Mean Absolute Error for the Model is')
    dis_col3.write(mean_absolute_error(Y,y_pred_rf_final))
    
    dis_col3.subheader('The Mean Squared Error for the Model is')
    dis_col3.write(mean_squared_error(Y,y_pred_rf_final))
