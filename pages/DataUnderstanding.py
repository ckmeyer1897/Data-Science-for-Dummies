import streamlit as st
from datetime import datetime
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from pages import utils
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st
import pandas_profiling
from pandas_profiling import ProfileReport

# from pages import pandasprofile
import streamlit_pandas_profiling

# from pandasprofile import st_profile_report
from streamlit_pandas_profiling import st_profile_report

st.set_option("deprecation.showPyplotGlobalUse", False)

# @st.cache()
# def get_summary_report(df_analysis):
#     pr = ProfileReport(df_analysis)
#     return pr


def app():
    # st.markdown("## Data Upload")

    # # Upload the dataset and save as csv
    # st.markdown("### Upload a csv file for analysis.") 
    # st.write("\n")

    # # # Code to read a single file 
    # # uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
    # # global data
    # if uploaded_file is not None:
    #     try:
    #         data = pd.read_csv(uploaded_file)
    #     except Exception as e:
    #         print(e)
    #         data = pd.read_excel(uploaded_file)

            
    st.markdown("## Data Exploration")
    st.write("\n")
    if "main_data.csv" not in os.listdir("data"):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        # df_analysis = pd.read_csv('data/2015.csv')
        df_analysis = pd.read_csv("data/main_data.csv")

        # #@st.cache
        # def get_summary_report(df_analysis):
        #     pr = ProfileReport(df_analysis)
        #     return pr

        if st.button('Generate Summary Report'):
            #pr = get_summary_report(df_analysis)
            pr = ProfileReport(df_analysis)
            st_profile_report(pr)

    # Code to read a single file
    # #    uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
    # global data
    # if uploaded_file is not None:
    #     try:
    #         data = pd.read_csv(uploaded_file)
    #     except Exception as e:
    #         print(e)
    #         data = pd.read_excel(uploaded_file)

    # # if st.button("Load Data"):
    # #     data  = pd.read_csv('bank-full.csv', delimiter = ';', nrows=1000)
    # #     # Raw data
    #     st.dataframe(data)
    #     data.to_csv('data/main_data.csv', index=False)
    #     numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    #     categorical_cols = list(set(list(data.columns)) - set(numeric_cols))
    #     columns = utils.genMetaData(data)
    #     columns_df = pd.DataFrame(columns, columns = ['column_name', 'type'])
    #     columns_df.to_csv('data/metadata/column_type_desc.csv', index = False)

    #     st.markdown("## Data Exploration")
    #     st.write('')
    #     """
    #     ##   Data Collection & Description
    #     """
    #     # Row 1 ----------------------------------------------------------------

    #     #pr = data.ProfileReport()
    #     pr = ProfileReport(data)
    #     st_profile_report(pr)

