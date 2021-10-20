# import libraries 
import os
import sys
import pandas as pd
import plotly
import streamlit as st
import plotly.express as px
import datetime

sys.path.insert(0, "..")

from Loading import loading
from Dashboard import categorical_cov_plots
from Dashboard import numerical_cov_plots
from Dashboard import concept_plots
from Dashboard import numerical_cov_plots
import constants as cst

class DashboardApp:
    def __init__(self, sample_df, batch_df):
        self.sample_df = sample_df 
        self.batch_df = batch_df
        
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
        option = st.sidebar.selectbox('Pick Dashboard:', ('Monitoring - Overview', 'Feature Distribution Analysis'))

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
        if self.option == 'Feature Distribution Analysis':
            st.subheader('Column Alerts')
            st.write('Add column x metrics alert matrix')

            st.subheader('Categorical Columns')
            fig_categorical_dist, fig_categorical_dist_diff = self.create_categorical_distribution_plots()
            st.plotly_chart(fig_categorical_dist)
            st.plotly_chart(fig_categorical_dist_diff)

            st.subheader('Numerical Columns')
            fig_numerical_dist = self.create_numerical_distribution_plots()
            st.plotly_chart(fig_numerical_dist)

    def create_categorical_distribution_plots(self, categorical_col="Education"):
        fig_categorical_dist = categorical_cov_plots.graph_categorical_dist(self.sample_df, self.batch_df, categorical_col)
        fig_categorical_dist_diff = categorical_cov_plots.graph_categorical_dist_diff(self.sample_df, self.batch_df, categorical_col) 
        return fig_categorical_dist, fig_categorical_dist_diff

    def create_numerical_distribution_plots(self, numerical_col="Income"):
        fig_numerical_dist = numerical_cov_plots.plot_distributions_numerical_variables(self.sample_df, self.batch_df, numerical_col)
        return fig_numerical_dist