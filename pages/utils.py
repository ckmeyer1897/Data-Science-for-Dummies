import numpy as np 
import pandas as pd 
from pandas.api.types import is_numeric_dtype
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from matplotlib.cbook import boxplot_stats  
from matplotlib.figure import Figure
import streamlit as st



# from pandas_profiling import ProfileReport

def isCategorical(col):
    unis = np.unique(col)
    if len(unis)<0.2*len(col):
        return True
    return False

# def getProfile(data):
#     report = ProfileReport(data)
#     report.to_file(output_file = 'data/output.html')

def getColumnTypes(cols):
    Categorical=[]
    Numerical = []
    Object = []
    for i in range(len(cols)):
        if cols["type"][i]=='categorical':
            Categorical.append(cols['column_name'][i])
        elif cols["type"][i]=='numerical':
            Numerical.append(cols['column_name'][i])
        else:
            Object.append(cols['column_name'][i])
    return Categorical, Numerical, Object

def isNumerical(col):
    return is_numeric_dtype(col)

def genMetaData(df):
    col = df.columns
    ColumnType = [] 
    Categorical = []
    Object = []
    Numerical = []
    for i in range(len(col)):
        if isCategorical(df[col[i]]):
            ColumnType.append((col[i],"categorical"))
            Categorical.append(col[i])
        
        elif is_numeric_dtype(df[col[i]]):
            ColumnType.append((col[i],"numerical"))
            Numerical.append(col[i])
        
        else:
            ColumnType.append((col[i],"object"))
            Object.append(col[i])

    return ColumnType

def makeMapDict(col): 
    uniqueVals = list(np.unique(col))
    uniqueVals.sort()
    dict_ = {uniqueVals[i]: i for i in range(len(uniqueVals))}
    return dict_

def mapunique(df, colName):
    dict_ = makeMapDict(df[colName])
    cat = np.unique(df[colName])
    df[colName] =  df[colName].map(dict_)
    return cat 

    
## For redundant columns
def getRedundentColumns(corr, y: str, threshold =0.1): 
    cols = corr.columns
    redunt = []
    k = 0
    for ind, c in enumerate(corr[y]):
        if c<1-threshold: 
            redunt.append(cols[ind])
    return redunt

def newDF(df, columns2Drop):
    newDF = df.drop(columns2Drop, axis = 'columns')
    return newDF

def summary_target(data, target):

    fig = Figure()
    ax = fig.subplots()
    sns.countplot(data = data, y = target)
    st.pyplot()
    plt.show();
    
#histogram of distribution 
def num_hist(data):
    data.select_dtypes(include=np.number).hist()
    plt.show();
    
# density plot of distribution
def num_dist(data):
    num_vars = []
    skews = []
    kurts = []
    dist_df = pd.DataFrame()
    for n in data.select_dtypes(include=np.number):
        sns.distplot(data[n])
        skew = data[n].skew()
        kurt = data[n].kurt()
        num_vars.append(n)
        skews.append(skew)
        kurts.append(kurt)
        plt.show();
        
    dist_df['Variable'] = num_vars
    dist_df['Skewness'] = skews
    dist_df['Kurtosis'] = kurts
    
    return dist_df

def num_anomal(dist_df):
    
    # Skewness above 3 is significant
    skewed = dist_df[dist_df['Skewness'] > 3]
    skew_vars = list(skewed['Variable'])

    # Excess Kurtosis above 5 is significant
    kurted = dist_df[abs((dist_df['Kurtosis'] > 5))] 
    kurted_vars = list(kurted['Variable'])
    
    return skew_vars, kurted_vars

#correlation matrix
def num_cmap(data):
    f, ax = plt.subplots(figsize=(9,6))
    sns.heatmap(data.select_dtypes(include=np.number).corr(), 
            vmin=-1, vmax=1, center=0, cmap='Blues', annot=True)
    plt.show();
    
    cm = data.corr()
    corrs = cm.stack().reset_index()
    corrs.columns = ['V1', 'V2', 'Corr']
    corrs['Abs Corr'] = abs(corrs['Corr'])
    corrs = corrs[corrs['Corr'] != 1]
    corrs = corrs.sort_values(by = 'Abs Corr', ascending=False)
    return corrs

def remove_correlated(data, correlations):
    v1 = list(set(correlations[correlations['Abs Corr'] > 0.7]['V1']))
    v2 = list(set(correlations[correlations['Abs Corr'] > 0.7]['V2']))
    correlated_vars = list(set(v1 + v2))

    data = data.drop(correlated_vars, axis=1)
    
    return data
    
def num_one_boxplt(data):
    """ Graphs all numerical variables on one axes"""
    num_cols = data.select_dtypes(include=['int','int64','float'])
    melted_df = pd.melt(num_cols)
    sns.set(rc={'figure.figsize':(6,4)})
    sns.boxplot(x='value',y='variable',data=melted_df)
    plt.show();

def num_multi_boxplt(data):
    """ Graphs all numerical variables on their own axis"""
    for i in data.columns:
        sns.set(rc={'figure.figsize':(6,4)})
        temp_df = data.loc[:,i]
        sns.boxplot(temp_df)
        plt.show();
        
def one_cat_counts(data):
    for category in data.select_dtypes(include='O'):
        sns.countplot(x=category, data=data)
        plt.show();
        
def bi_cat_counts(data, target):
    for category in data.select_dtypes(include='O'):
        sns.countplot(y=category, data=data, hue = target, orient = 'v')
        plt.show();
sns.set_palette('pastel')

# density plot of distribution
def num_dist_targ(data, target):
    """ Plots all the numerical variables with the target variable overlaid. Because there are
    many more nos then yes's we normalized the graph to have a clearer picture of the graph
    """
    for n in data.select_dtypes(include=np.number):
        sns.displot(data, x = data[n], hue= target, element = 'step', stat='density', common_norm = False)
        plt.show();
  
def split_data(data):
    num_data = data.select_dtypes(include=np.number)
    cat_data = data.select_dtypes(include='O')
    return num_data, cat_data

def load_data(nrows):
    data  = pd.read_csv('bank-full.csv', delimiter = ';', nrows=nrows)
    return data

def load_df(df_name):
    data  = pd.read_csv('data/'+ df_name + '.csv')
    return data

if __name__ == '__main__':
    df = {"Name": ["salil", "saxena", "for", "int"]}
    df = pd.DataFrame(df)
    print("Mapping dict: ", makeMapDict(df["Name"]))
    print("original df: ")
    print(df.head())
    pp = mapunique(df, "Name")
    print("New df: ")
   # print(pp.head())


