# import python libraries
import pandas as pd
import streamlit as st

@st.cache
def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df


# pos display

# NER display


# topic modeling display

# text classification model inference