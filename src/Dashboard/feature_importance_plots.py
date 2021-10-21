
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import pandas as pd
import plotly.graph_objects as go

from Evaluation import feature_importance
from Utils import utils as u
import constants as cst

def graph_feature_importance(sample_df: pd.DataFrame):
    """Plot feature importance 

    Args:
        sample_df (pd.DataFrame): training data
    """
    importances_df = get_feature_importance_to_plot(sample_df)
    importances_df = importances_df.sort_values('importance_score', ascending=False)
    fig = go.Figure(
        go.Bar(
            x=importances_df.sort_values('importance_score', ascending=False).index, 
            y=importances_df.sort_values('importance_score', ascending=False)['importance_score']
        )
    )
    return fig

def get_feature_importance_to_plot(sample_df):
    model = u.load_model()
    importances_df = feature_importance.extract_feature_importance(
        model, 
        sample_df.drop(columns=[cst.y_name]), 
        sample_df[cst.y_name]
    )
    return importances_df.sort_values('importance_score', ascending=False)
