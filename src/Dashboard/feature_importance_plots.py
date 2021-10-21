
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
    X_train = sample_df.copy()
    X_train = X_train.drop(
        columns=[
        cst.y_name, 
        cst.y_pred, 
        f"{cst.y_pred_proba}_{cst.y_class_labels[0]}", 
        f"{cst.y_pred_proba}_{cst.y_class_labels[1]}"
        ]
    )
    y_train = sample_df[cst.y_name]

    importances_df = feature_importance.extract_feature_importance(
        model, 
        X_train, 
        y_train
    )
    return importances_df.sort_values('importance_score', ascending=False)
