import streamlit as st
import numpy as np
import pandas as pd
import pages
from pages import utils
#from utils import summary_target
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.figure import Figure
import random
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from pages.utils import summary_target



def app():
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        data = pd.read_csv('data/main_data.csv')

## 3.1 Downsampling
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('1 - Downsampling Data')
        @st.cache
        def downsample(data):
            majority_class = data.loc[data['y']=='no',:]
            minority_class = data.loc[data['y']=='yes',:]

            random_index = random.sample(majority_class.index.tolist(), data['y'].value_counts()[1])
            majority_class = majority_class.loc[random_index,:]
            balanced_df = pd.concat([majority_class,minority_class],axis=0)

            return balanced_df

        if st.button("Downsample Data"):
            balanced_df = downsample(data)
            st.write(balanced_df.shape)
            st.write(balanced_df['y'].value_counts())
            balanced_df.to_csv('data/balanced_data.csv', index=False)
            summary_target(balanced_df, 'y')

## 3.2 Clean Data
    with col2:
        st.subheader('2 - Clean Data')
        def remove_outliers(data):
            for i in data.select_dtypes(include=np.number):
                Q1 , Q3 = data[i].quantile(0.25) , data[i].quantile(0.75)
                IQR = Q3 - Q1 #Interquartile range
                FQR, TQR = Q1 - 1.5 * IQR , Q3 + 1.5 *IQR
                data = data[(data[i] >= FQR) & (data[i] <= TQR)]
                return data
        
        def remove_correlated(data, threshold):
            cm = data.corr()
            corrs = cm.stack().reset_index()
            corrs.columns = ['V1', 'V2', 'Corr']
            corrs['Abs Corr'] = abs(corrs['Corr'])
            corrs = corrs[corrs['Corr'] != 1]
            
            v1 = list(set(corrs[corrs['Abs Corr'] > 0.7]['V1']))
            v2 = list(set(corrs[corrs['Abs Corr'] > 0.7]['V2']))
            correlated_vars = list(set(v1 + v2))

            data = data.drop(correlated_vars, axis=1)
            
            return data, correlated_vars

        if st.button("Remove Outliers"):
            balanced_df = pd.read_csv('data/balanced_data.csv')
            no_outliers_df = remove_outliers(balanced_df)
            no_outliers_df.to_csv('data/no_outliers.csv', index=False)
            st.write(no_outliers_df.shape)

        if st.button("Remove Correlated"):
            threshold = st.multiselect('Select Correlation Threshold', [0,.1,.2,.3,.4,.5,.6,.7,.8,.9])
            no_out = pd.read_csv('data/no_outliers.csv')
            if threshold is not None:
                no_out_corr, correlated = remove_correlated(no_out, threshold)
                st.write(no_out_corr.shape)
                st.write('Correlated Variables', correlated)
                no_out_corr.to_csv('data/clean_df.csv', index=False)
    
        
            

## 3.3 Feature Engineering
 ### Functions
    def split_cats(cat_data):
        """ Splits categorical features into binary and multi"""
        cat_counts = cat_data.describe().loc['unique']
        multi_cats = cat_counts[cat_counts > 2]
        bi_cats = cat_counts[cat_counts <= 2]
        multi_cats_list = list(multi_cats.index)
        bi_cats_list = list(bi_cats.index)

        return bi_cats_list, multi_cats_list

    def get_binary(cat_data, bi_cats):
        """ Converts all discrete categorical variables to binary"""
        ordinal_status = {'yes': 1, 'no': 0}
        binary_data = cat_data.loc[:,bi_cats]
        for i in bi_cats:
            binary_data[i].replace(ordinal_status, inplace = True)
        return binary_data

    def transform_cat(data):
        cat_data = data.select_dtypes(include='O')
        # get multi and binary categorical variables
        bi_cats, multi_cats = split_cats(cat_data)
        #get dummy variables for nomial and ordinal categories 
        nom_data = pd.get_dummies(cat_data.loc[:,multi_cats])
        ordinal_data = get_binary(cat_data, bi_cats)
        return nom_data, ordinal_data

    def integrate_data(data, ordinal_data, nom_data):
        # join the ordinal features with the dummy features to form categorical data
        num_data = data.select_dtypes(include=np.number)
        cat_data = pd.concat([ordinal_data, nom_data], axis = 1)

        df = pd.concat([cat_data, num_data], axis = 1)
        print(df.shape)
        #print(list(df.columns))
        return df
 ### Code
    clean_df = pd.read_csv('data/clean_df.csv')
    #st.write('Clean Data:' ,clean_df)

    with col1:
        st.subheader('3- Feature Engineering')
        nom_data, ordinal_data = transform_cat(clean_df)
        if st.button("Transform Cateogricals (Dummy Variables)"):
            st.write('Nomial Data:' , nom_data)
            st.write('Ordinal Data', ordinal_data)

    with col2:
        st.subheader('4 - Integrate Data')
        if st.button('Merge Numercail & Cateogrical'):
            integrated_df = integrate_data(clean_df, ordinal_data, nom_data)
            st.write('Integrated Data', integrated_df)
            integrated_df.to_csv('data/integrated_df.csv', index= False)
    
## 3.5 Format Data
    def normalize_df(df):
        sc = MinMaxScaler(feature_range=(0,1))
        scaled_balanced_df = sc.fit_transform(df)
        scaled_balanced_df = pd.DataFrame(scaled_balanced_df, index=df.index, columns=df.columns)
        return scaled_balanced_df

    col1, col2 = st.columns(2)
    integrated_df = pd.read_csv('data/integrated_df.csv')

    with col1:
        st.subheader('5 - Normalize Data')
        if st.button('Min Max Scaler'):
            scaled_balanced_df = normalize_df(integrated_df)
            scaled_balanced_df.to_csv('data/df.csv', index=False)
            st.write('Processed Data: ', scaled_balanced_df)

    
