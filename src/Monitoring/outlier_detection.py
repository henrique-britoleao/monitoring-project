import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px


def detect_outliers(
    df_preprocessed: pd.DataFrame, threshold: float = -0.075
) -> pd.DataFrame:
    """ Detects outliers using the Isolation Forest algorithm and outputs a
    dataframe with an outlier score and anomaly label.

    Args:
        df_preprocessed (pd.DataFrame): preprocessed data
        threshold (float): threshold to classify anomaly label based on
        outlier score

    Returns:
        df_outlier (pd.DataFrame): outlier score and anomaly label

    """
    df_outlier = pd.DataFrame()
    rng = np.random.RandomState(42)

    clf = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        bootstrap=False,
        n_jobs=-1,
        random_state=rng,
    )
    clf.fit(df_preprocessed)

    df_outlier["scores"] = clf.decision_function(df_preprocessed)
    df_outlier["anomaly"] = df_outlier["scores"].apply(
        lambda x: "outlier" if x <= threshold else "inlier"
    )

    return df_outlier


def compute_outlier_percentage(df_preprocessed: pd.DataFrame) -> float:
    """Compute outlier percentage of dataframe in json file.

    Args:
        df_preprocessed (pd.DataFrame): preprocessed data

    Returns:
        outlier_percentage (float): outlier percentage in preprocessed dataset
    """
    df_outliers = detect_outliers(df_preprocessed)
    n_rows = df_outliers.shape[0]
    n_outliers = df_outliers[df_outliers["anomaly"] == "outlier"].shape[0]
    outlier_percentage = round(n_outliers / n_rows, 4)

    return outlier_percentage


def build_outlier_dict(df_preprocessed: pd.DataFrame) -> dict:
    """Build outliers dict for batch analysis.

    Args:
        df_preprocessed (pd.DataFrame): preprocessed data

    Returns:
        outliers (dict): percentage outlier in dict
    """
    outlier_percentage = compute_outlier_percentage(df_preprocessed)
    outliers = {"percentage_outlier": outlier_percentage}

    return outliers


def plot_outliers(df_preprocessed: pd.DataFrame, path: str = None) -> px.Figure:
    """Save outlier plot from output dataframe of detect_outliers function.

    Args:
        df_preprocessed (pd.DataFrame): preprocessed data
        path (string): path and name to save plot

    Returns:
        fig (Figure): plot of outlier and inlier from data
    """
    df_outliers = detect_outliers(df_preprocessed)
    fig = px.histogram(
        df_outliers,
        x="scores",
        color="anomaly",
        color_discrete_map={"outlier": "rgb(0, 0, 100)", "inlier": "rgb(0, 200, 200)"},
    )

    if path:
        fig.write_image(path)

    return fig
