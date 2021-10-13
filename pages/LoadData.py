import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
import pandas_profiling
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


#@st.cache
def app():
    st.markdown("## Data Upload")

    # Upload the dataset and save as csv
    st.markdown("### Upload a csv file for analysis.") 
    st.write("\n")

    # Code to read a single file 
    uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx', 'pickle'])
    global data
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            data = pd.read_excel(uploaded_file)
    

    ''' Load the data and save the columns with categories as a dataframe. 
    This section also allows changes in the numerical and categorical columns. '''
    if st.button("Load Data"):
     #   st.progress(0)
        data  = pd.read_csv('bank-full.csv', delimiter = ';', nrows=1000)
      #  st.progress(100)

        # Raw data 
       # st.dataframe(data)
        data.to_csv('data/main_data.csv', index=False)
        
        @st.cache
        def get_summary_report(df_analysis):
            pr = ProfileReport(df_analysis)
            return pr

        #Generate a pandas profiling report
        if st.button("Generate an analysis report"):
            pr = get_summary_report(data)
            st_profile_report(pr)
        #    utils.getProfile(data)
            #Open HTML file

        #pass
        st.dataframe(data)  
        # Collect the categorical and numerical columns 
        
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = list(set(list(data.columns)) - set(numeric_cols))
        
        # Save the columns as a dataframe or dictionary
        columns = []

        # Iterate through the numerical and categorical columns and save in columns 
        columns = utils.genMetaData(data) 
        
        # Save the columns as a dataframe with categories
        # Here column_name is the name of the field and the type is whether it's numerical or categorical
        columns_df = pd.DataFrame(columns, columns = ['column_name', 'type'])
        columns_df.to_csv('data/metadata/column_type_desc.csv', index = False)
