import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from AutoML import num_hist, num_cmap, num_dist_targ, split_data, num_dist, load_data
from matplotlib.figure import Figure
st.set_option('deprecation.showPyplotGlobalUse', False)


st.set_page_config(page_title="Data Mining WorkFlow Tool", 
                   page_icon=":notes:", 
                   layout='wide')




row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('Data Mining WorkFlow Tool')
with row0_2:
    st.text("")
    st.subheader('Streamlit App by [Christian Meyer](https://www.linkedin.com/in/christian-meyer-1a0b21116/)')
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))

st.sidebar.header('Data Mining Phases')

"""
#  Data Understanding
"""
st.write('')
"""
##  Data Collection & Description
"""
# Row 1 ----------------------------------------------------------------

data_load_state = st.text('Loading data...')
data = load_data(5000)
data_load_state.text("Done! (using st.cache)")
st.write('')
row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns(
    (.15, 1, .3, 4, .00000001))
  
with row1_1:
    st.subheader('Summary Info ')
    st.text(f"Columns : { str(data.shape[1])}")
    st.text(f"Rows : { str(data.shape[0])}")

with row1_2:
    st.dataframe(data.head())

features = list(data.columns)
num_data , cat_data = split_data(data)

"""
##  Data Exploration
"""
sns.set_palette('pastel')
# Row 2 ----------------------------------------------------
row2_space1, row2_1, row2_space2, row2_2, row2_space3, row2_3, row2_space4 = st.columns(
    (.15, 1.5, .00000001, 1.5, .00000001, 1.5, 0.15))

with row2_1:
    feature = st.selectbox('Select Feature', features)

with row2_3:
    #target = st.selectbox('Target Variable', features['y'])
    target = 'y'
    st.text("Y is the target")

#Row 3 ---------------------------------------------------------------------
st.write('')
row2_space1, row2_1, row2_space2, row2_2, row2_space3, row2_3, row2_space4 = st.columns(
    (.15, 1.5, .00000001, 1.5, .00000001, 1.5, 0.15))

with row2_1:
    st.subheader('Feature Distribution')
    fig = Figure()
    ax = fig.subplots()
    sns.histplot(data[feature], ax = ax , kde= True)
    st.pyplot(fig)


with row2_2:
    st.subheader('Target Variable')
    fig = Figure()
    ax = fig.subplots()
    sns.countplot(data[target])
    st.pyplot()

with row2_3:
    st.subheader('Target Variabel Overlay')
    fig = Figure()
    ax = fig.subplots()
    if feature in list(num_data.columns):
       sns.displot(data, x = data[feature],hue = target, stat = 'density', common_norm =False)
    else:
        sns.countplot(y = data[feature], data = data, hue = target, orient='v')   
    st.pyplot()

"""
### Multivariate Analysis
"""
st.write('')
row4_space1, row4_1, row4_space2, row4_2, row4_space3, row4_3, row4_space4 = st.columns(
    (.15, 1.5, .00000001, 1.5, .00000001, 1.5, 0.15))

with row4_1:
    st.subheader('Correlation')
    f = Figure()
    ax = f.subplots()
    f, ax = plt.subplots()
    st.write(sns.heatmap(data.select_dtypes(include=np.number).corr(), vmin=-1, vmax=1, center=0, cmap='Blues', annot=True))
    st.pyplot()







