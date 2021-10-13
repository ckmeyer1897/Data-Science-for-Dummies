import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.figure import Figure


def app():
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        # df_analysis = pd.read_csv('data/2015.csv')
        df_analysis = pd.read_csv('data/main_data.csv')
        # df_visual = pd.DataFrame(df_analysis)
        df_visual = df_analysis.copy()
        cols = pd.read_csv('data/metadata/column_type_desc.csv')
        Categorical,Numerical,Object = utils.getColumnTypes(cols)
        cat_groups = {}
        unique_Category_val={}

        for i in range(len(Categorical)):
                unique_Category_val = {Categorical[i]: utils.mapunique(df_analysis, Categorical[i])}
                cat_groups = {Categorical[i]: df_visual.groupby(Categorical[i])}
                
        features = list(df_analysis.columns)
        feature = st.selectbox("Select Category ", features)
        target = st.selectbox("Select Target ", 'y')
       # target = st.text_input('Target Variable')
        sizes = (df_visual[feature].value_counts()/df_visual[feature].count())
        num_data = df_visual.select_dtypes(include=np.number)


        labels = sizes.keys()
        maxIndex = np.argmax(np.array(sizes))
        explode = [0]*len(labels)
        explode[int(maxIndex)] = 0.1
        explode = tuple(explode)

        #df_grouped = df_analysis.groupby(feature).count()[target]
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader('Feature Distribution')
            fig = Figure()
            ax = fig.subplots()
            sns.histplot(df_visual[feature], ax = ax , kde= True)
            st.pyplot(fig)

        with col2: 
            st.subheader('Skewness')
            skew = df_analysis[feature].skew()
            kurt = df_analysis[feature].kurt()
            st.write("Skew", skew)
            st.write('Kurt: ', kurt)
        
        with col3:
            st.subheader('Target Variabel Overlay')
            fig = Figure()
            ax = fig.subplots()
            if feature in list(num_data.columns):
                sns.displot(data = df_visual, x = df_visual[feature],hue = target, stat = 'density', common_norm =False)
            else:
                sns.countplot(y = df_visual[feature], data = df_visual, hue = target, orient='v')   
            st.pyplot()

            # Row 2 ----------------------------------------------------

        # row0_spacer1, row2_1, row0_spacer2, row2_2, row0_spacer3 = st.columns(
        #     (.1, 2, .1, 2, .1))
        # # with row2_1:
        # #     st.subheader('Feature Distribution')
        # #     fig = Figure()
        #     ax = fig.subplots()
        #     sns.histplot(df_visual[feature], ax = ax , kde= True)
        #     st.pyplot(fig)
        
        # corr = df_analysis.corr(method='pearson')
        # with row2_2:
        #     fig2, ax2 = plt.subplots()
        #     mask = np.zeros_like(corr, dtype=np.bool)
        #     mask[np.triu_indices_from(mask)] = True
        #     # Colors
        #     cmap = sns.diverging_palette(240, 10, as_cmap=True)
        #     sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, center=0,ax=ax2)
        #     ax2.set_title("Correlation Matrix")
        #     st.pyplot(fig2)
            
        # target = st.selectbox("Select Target ", features)
        # row0_spacer1, row3_1, row0_spacer2, row3_2, row0_spacer3 = st.columns(
        #     (.1, 2, .1, 2, .1))

        # with row3_1:
        #     st.subheader('Target Variable')
        #     fig, ax = plt.subplots()
        #     sns.countplot(data = df_visual, x = df_visual[target] ,ax = ax)
        #     st.pyplot()

        # with row3_2:
        #     st.subheader('Target Variabel Overlay')
        #     fig = Figure()
        #     ax = fig.subplots()
        #     if feature in list(num_data.columns):
        #         sns.displot(data = df_visual, x = df_visual[feature],hue = target, stat = 'density', common_norm =False)
        #     else:
        #         sns.countplot(y = df_visual[feature], data = df_visual, hue = target, orient='v')   
        #     st.pyplot()
