import os
import streamlit as st
import numpy as np
from PIL import  Image
import pandas as pd
from streamlit.elements import image


# Custom imports 
from multipage import MultiPage
from pages import LoadData, DataUnderstanding, data_viz, dataprep, modeling

# Create an instance of the app 
app = MultiPage()

st.set_page_config(page_title="Data Mining", 
                   page_icon=":notes:", 
                   layout='wide')


# Title of the main page
row0_spacer1, row0_1, row1_spacer1, row0_2, row0_spacer3  = st.columns((.1, 2, .1, 6,.1))
display = Image.open('logo.png')
display = np.array(display)
row0_1.image(display, width =200)

with row0_2:
    st.title('Data Voyager')
    st.subheader('Automating Machine Learning Pipelines')

row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.markdown('##### Created by [Christian Meyer](https://www.linkedin.com/in/christian-meyer-1a0b21116/)')
st.markdown('***')

# Add all your application here
app.add_page("Load Data", LoadData.app)
app.add_page("Data Exploration & Vizualization", DataUnderstanding.app)
#app.add_page("Data Vizualization", data_viz.app)
app.add_page("Data Preparation", dataprep.app)
app.add_page("Modeling", modeling.app)





# The main app
app.run()