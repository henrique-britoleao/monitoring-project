# import libraries 
import os
import pandas as pd
import plotly
import streamlit as st
import plotly.express as px
import datetime

class DashboardApp:
    def __init__(self, sample_df, batch_dfs):
        self.sample_df = sample_df 
        self.batch_dfs = batch_dfs
        
        # to be created 
        self.option = None
    
    def configure_page(self):
        '''
        Configures app page
        Creates sidebar with selectbox leading to different main pages 
        
        Returns:
            option (str): Name of main page selected by user
        '''
        st.set_page_config(
             page_title="Continuous Monitoring",
             layout="wide",
             initial_sidebar_state="expanded",
        )

        # create sidebar
        st.sidebar.title("Model Monitoring")
        option = st.sidebar.selectbox('Pick Dashboard:', ('Monitoring - Overview', 'Categorical Columns', 'Numerical Columns'))

        self.option = option


    def create_main_pages(self):
        '''
        Creates pages for all options available in sidebar
        Page is loaded according to option picked in sidebar
        '''

        # Main Dashboard
        if self.option == 'Monitoring - Overview':
            st.title("Monitoring Overview")

            st.subheader('Project Data')
            st.dataframe(self.sample_df.head(5))
            # Placeholder: Response Distribution

            st.subheader('Model Performance Evaluation')
            st.markdown("Evaluating the performance of our classification model over time.")
            # Placeholder: ROC/AUC Curve, Main classification metrics (training vs batch x, y, z)

            st.subheader('Streaming Data Evolution')
            st.markdown("Identifying potential concept drift. ")
            # Placeholder: Data description (initial vs batch).
            # Placeholder: Raised alerts 

        # Categorical Columns
        if self.option == 'Categorical Columns':
            # placeholder: viz distribution categorical columns (sample vs batch)

        # Numerical Columns 
        if self.option == 'Numerical Columns':
            # placeholder: viz distribution numerical columns (sample vs batch), cumulative distribution, mean diff etc 

