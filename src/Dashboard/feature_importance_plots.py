# -*- coding: utf-8 -*-

import pandas as pd
import plotly.graph_objects as go

from evaluation import feature_importance
from utils import utils as u
import constants as cst

#####  Set Logger  #####
from src.utils.loggers import MainLogger

logger = MainLogger.getLogger(__name__)


def graph_feature_importance(
    sample_df: pd.DataFrame, colors: list = ["rgb(0, 0, 100)", "rgb(0, 200, 200)"]
):
    """Plot feature importance 

    Args:
        sample_df (pd.DataFrame): training data
    """
    importances_df = get_feature_importance_to_plot(sample_df)
    importances_df = importances_df.sort_values("importance_score", ascending=False)
    fig = go.Figure(
        go.Bar(
            x=importances_df.sort_values("importance_score", ascending=False).index,
            y=importances_df.sort_values("importance_score", ascending=False)[
                "importance_score"
            ],
            marker_color=colors[0],
        )
    )
    return fig


def get_feature_importance_to_plot(sample_df):
    """Compute feature importance using permutation importance on the selected model and the training data

    Args:
        sample_df (pd.DataFrame): training data

    Returns:
        pd.DataFrame: dataframe with features as index and importance scores as values
    """
    model = u.load_model()
    X_train = sample_df.copy()
    for col in [
        cst.y_name,
        cst.y_pred,
        f"{cst.y_pred_proba}_{cst.y_class_labels[0]}",
        f"{cst.y_pred_proba}_{cst.y_class_labels[1]}",
    ]:
        if col in X_train.columns:
            X_train = X_train.drop(col, axis=1)
    y_train = sample_df[cst.y_name]

    importances_df = feature_importance.extract_feature_importance(
        model, X_train, y_train
    )
    return importances_df.sort_values("importance_score", ascending=False)
