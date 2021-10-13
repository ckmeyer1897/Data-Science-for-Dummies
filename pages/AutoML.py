import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from matplotlib.cbook import boxplot_stats  
import numpy as np
import pandas as pd



def summary_target(data, target):
    summary = target.describe()
    true_count = target.value_counts()
    true_rate = true_count[1] / true_count[0]
    print('Summary Info:')
    print(summary)
    print('')
    print("Percent Positive: " +  str(true_rate *100))
    
    sns.countplot(target)
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