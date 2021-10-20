
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import pandas as pd
import plotly.graph_objects as go

def graph_feature_importance(importances_df: pd.DataFrame):
    """Plot feature importance 

    Args:
        importances_df (pd.DataFrame): dataframe with features as index and importance score as value
    """
    importances_df = importances_df.sort_values('importance_score', ascending=False)
    fig = go.Figure(
        go.Bar(
            x=importances_df.sort_values('importance_score', ascending=False).index, 
            y=importances_df.sort_values('importance_score', ascending=False)['importance_score']
        )
    )
    return fig