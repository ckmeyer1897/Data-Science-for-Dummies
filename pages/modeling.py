import streamlit as st
import numpy as np
import pandas as pd
#import pages
import utils
from utils import load_df
#from utils import summary_target
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.figure import Figure
import random
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
#from pages.utils import summary_target

def app():

    df = pd.read_csv('data/df.csv')

    df2 = load_df('df')

    st.write(df2)