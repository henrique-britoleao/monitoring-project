import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest


def detect_outliers(
    df_preprocessed: pd.DataFrame, threshold: float = -0.05
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


def compute_outlier_percentage(df_preprocessed: pd.DataFrame, path: str) -> float:
    """Save outlier percentage of dataframe in json file.

    Args:
        df_preprocessed (pd.DataFrame): preprocessed data
        path (string): path and name to save plot

    Returns:
        fig (Figure): plot of outlier and inlier from data
    """
    df_outliers = detect_outliers(df_preprocessed)
    n_rows = df_outliers.shape[0]
    n_outliers = df_outliers[df_outliers["anomaly"] == "outlier"].shape[0]
    outlier_percentage = 100 * n_outliers / n_rows

    return outlier_percentage


def plot_outliers(df_preprocessed: pd.DataFrame, path: str = None) -> sns.Figure:
    """Save outlier plot from output dataframe of detect_outliers function.

    Args:
        df_preprocessed (pd.DataFrame): preprocessed data
        path (string): path and name to save plot

    Returns:
        fig (Figure): plot of outlier and inlier from data
    """
    df_outliers = detect_outliers(df_preprocessed)
    outlier_plot = sns.histplot(x="scores", data=df_outliers, hue="anomaly")
    fig = outlier_plot.get_figure()
    if path:
        fig.savefig(path)
    return fig
