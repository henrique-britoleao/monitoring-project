# import libraries
import os
import sys
import pandas as pd
import plotly
import streamlit as st
import plotly.express as px
import datetime

from evaluation import feature_importance

sys.path.insert(0, "..")

import loading
from dashboard import model_perf_graph
from dashboard import concept_plots
from dashboard import feature_importance_plots
from dashboard import categorical_cov_plots
from dashboard import numerical_cov_plots
from dashboard import numerical_cov_plots
from dashboard import alert_plots
from dashboard import show_logs
from preprocessing import preprocessing
from main import main
import constants as cst


class DashboardApp:
    def __init__(self, sample_df):
        self.sample_df = sample_df
        self.batch_df = None
        self.batch_name = None
        self.batch_id = 1
        # TODO
        self.option = None

    def configure_page(self):
        """
        Configures app page
        Creates sidebar with selectbox leading to different main pages

        Returns:
            option (str): Name of main page selected by user
        """
        st.set_page_config(
            page_title="Continuous Monitoring",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # create sidebar
        st.sidebar.title("Model Monitoring")
        option = st.sidebar.selectbox(
            "Pick Dashboard:",
            (
                "Monitoring - Overview",
                "Model Performance Analysis",
                "Feature Distribution Analysis",
                "Concept Drift Analysis",
            ),
        )
        self.option = option
        uploaded_file = st.sidebar.file_uploader("Choose a file")
        if uploaded_file is not None:
            self.batch_df = pd.read_csv(uploaded_file, sep=None, engine="python")
            self.batch_name = cst.BATCH_NAME_TEMPLATE.substitute(id=self.batch_id)
            loading.write_csv_from_path(
                self.batch_df, os.path.join(cst.BATCHES_PATH, self.batch_name)
            )
            self.batch_df = batch_preprocess(
                self.batch_df, cst.column_types, preprocessing.MarketingPreprocessor()
            )
            self.batch_id += 1  # increment counter

    def create_main_pages(self):
        """
        Creates pages for all options available in sidebar
        Page is loaded according to option picked in sidebar
        """
        # Main Dashboard
        if self.option == "Monitoring - Overview" and self.batch_df is not None:
            st.title("Monitoring Overview")

            st.subheader("Project Data")
            st.dataframe(self.sample_df.head(5))
            # Placeholder: Response Distribution

            st.subheader("Model Performance Evaluation")
            st.markdown(
                "Evaluating the performance of our classification model over time."
            )
            # Placeholder: ROC/AUC Curve, Main classification metrics (training vs batch x, y, z)

            st.subheader("Streaming Data Evolution")
            st.markdown("Identifying potential concept drift. ")
            with show_logs.st_stderr("code"):
                main(self.batch_id - 1)
                graph = alert_plots.alerts_graph(self.batch_name, self.batch_id - 2)
                st.graphviz_chart(graph)

            # Placeholder: Data description (initial vs batch).
            # Placeholder: Raised alerts

        # Model Performance Analysis
        if self.option == "Model Performance Analysis" and self.batch_df is not None:
            st.subheader("Model performance evolution")
            st.write("Visualize classification performance metrics (cross-validation scores using the training data vs. true performance on the batch data:")
            fig_model_performance = model_perf_graph.plot_performance(
                batch_name=self.batch_name,
                batch_perf_path=cst.PERFORMANCE_METRICS_FILE_PATH,
                train_perf_path=cst.TRAIN_PERFORMANCE_METRICS_FILE_PATH,
            )

            st.plotly_chart(fig_model_performance)
            st.subheader(f"Feature importance for selected model {cst.selected_model}")
            st.write("Visualize the relative feature importance of each feature used in the model using permutation importance:")
            fig_feature_importance = self.create_feature_importance_plot()
            st.plotly_chart(fig_feature_importance)

        # Feature Distribution Analysis
        if self.option == "Feature Distribution Analysis" and self.batch_df is not None:
            st.title("Feature Distribution Analysis")
            st.subheader("Column Alerts")
            fig_heatmap = alert_plots.alerts_matrix(self.batch_name, self.batch_id - 2)
            st.plotly_chart(fig_heatmap)

            st.subheader("Numerical Columns")
            fig_numerical_scaled_means = (
                self.create_numerical_distribution_plots_all_cols()
            )
            st.plotly_chart(fig_numerical_scaled_means)

            numerical_column = st.selectbox(
                "Select numerical column to deep dive", cst.numerical_columns
            )
            (
                fig_numerical_boxplot,
                fig_numerical_dist,
            ) = self.create_numerical_distribution_plots(numerical_column)
            st.plotly_chart(fig_numerical_boxplot)
            st.plotly_chart(fig_numerical_dist)

            st.subheader("Categorical Columns")
            categorical_column = st.selectbox(
                "Select categorical column", cst.categorical_columns
            )
            (
                fig_categorical_dist,
                fig_categorical_dist_diff,
            ) = self.create_categorical_distribution_plots(categorical_column)
            st.plotly_chart(fig_categorical_dist)
            st.plotly_chart(fig_categorical_dist_diff)

        # Concept Drift Analysis
        if self.option == "Concept Drift Analysis" and self.batch_df is not None:
            st.subheader(f"Drift in predictions when using {cst.selected_model}")
            st.write("Change in distribution of target probability predictions")
            fig_cov_drift = concept_plots.graph_target_prob_dist(
                cst.PREDICTED_TRAIN_FILE_PATH, cst.PREDICTED_TRAIN_FILE_PATH
            )
            st.plotly_chart(fig_cov_drift)

    def create_categorical_distribution_plots(self, categorical_col="Education"):
        fig_categorical_dist = categorical_cov_plots.graph_categorical_dist(
            self.sample_df, self.batch_df, categorical_col
        )
        fig_categorical_dist_diff = categorical_cov_plots.graph_categorical_dist_diff(
            self.sample_df, self.batch_df, categorical_col
        )
        return fig_categorical_dist, fig_categorical_dist_diff

    def create_numerical_distribution_plots_all_cols(self):
        fig_numerical_scaled_means = numerical_cov_plots.plot_scaled_means(
            self.sample_df, self.batch_df
        )
        return fig_numerical_scaled_means

    def create_numerical_distribution_plots(self, numerical_col="Income"):
        fig_numerical_boxplot = numerical_cov_plots.plot_quartiles_numerical_variables(
            self.sample_df, self.batch_df, numerical_col
        )
        fig_numerical_dist = numerical_cov_plots.plot_distributions_numerical_variables(
            self.sample_df, self.batch_df, numerical_col
        )
        return fig_numerical_boxplot, fig_numerical_dist

    def create_feature_importance_plot(self):
        fig_feature_importance = feature_importance_plots.graph_feature_importance(
            self.sample_df
        )
        return fig_feature_importance
        pass


def batch_preprocess(
    batch_df: pd.DataFrame, column_types, preprocessor: preprocessing.Preprocessor
):
    return preprocessor(batch_df, column_types)
