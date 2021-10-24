# -*- coding: utf-8 -*-

import pandas as pd
import plotly.graph_objects as go

from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.inspection import permutation_importance

#####  Set Logger  #####
from src.utils.loggers import MainLogger
logger = MainLogger.getLogger(__name__)

#TODO: clf to pipeline
def extract_feature_importance(clf: ClassifierMixin, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Compute feature importance using permutation importance method
    Args:
        clf: model pipeline
        X_train (pd.DataFrame): X_train
        y_train (pd.DataFrame): y_train

    Returns: Dict of feature importance values
    """
    clf2 = clone(clf)
    clf2.fit(X_train, y_train)
    importances = permutation_importance(clf2, X_train, y_train,
                            n_repeats=5,
                           random_state=42)

    importances_df = pd.DataFrame(
        index=X_train.columns, 
        data={'importance_score': importances['importances_mean']}
    )

    return importances_df

def plot_feature_importance(importances_df: pd.DataFrame):
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
    fig.show()