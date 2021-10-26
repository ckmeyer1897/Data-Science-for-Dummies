import os
import streamlit as st
import numpy as np
from PIL import  Image
import pandas as pd


# Custom imports 
from multipage import MultiPage
from pages import LoadData, DataUnderstanding, utils, data_viz, dataprep

# Create an instance of the app 
app = MultiPage()

st.set_page_config(page_title="Data Understanding", 
                   page_icon=":notes:", 
                   layout='wide')
# Title of the main page
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('Data Mining WorkFlow Tool')
with row0_2:
    st.text("")
    st.subheader('Streamlit App by [Christian Meyer](https://www.linkedin.com/in/christian-meyer-1a0b21116/)')
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))

 #st.image(display, width = 400)
# st.title("Data Storyteller Application")
#col1, col2 = st.columns(2)
#col1.image(display, width = 400)
st.sidebar.title("1 - Data Understanding")#
# with st.sidebar.beta_expander(app.add_page('Data Collection', LoadData.app)
# Add all your application here
app.add_page("Load Data", LoadData.app)
app.add_page("Data Exploration", DataUnderstanding.app)
app.add_page("Data Vizualization", data_viz.app)
app.add_page("Data Preparation", dataprep.app)


# st.sidebar.title("2 - Data Preparation")
# st.sidebar.title("3 - Modeling")
# st.sidebar.title("4 - Evaluation")





# The main app
app.run()