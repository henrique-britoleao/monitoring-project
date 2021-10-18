import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest


def detect_outliers(df: pd.DataFrame, threshold=-0.05):
    """ Detects outliers using the Isolation Forest algorithm and outputs a
    dataframe with an outlier score and anomaly label.

    Args:
        df (pd.DataFrame): preprocessed data
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
    clf.fit(df)

    df_outlier["scores"] = clf.decision_function(df)
    df_outlier["anomaly"] = df_outlier["scores"].apply(
        lambda x: "outlier" if x <= threshold else "inlier"
    )

    return df_outlier


def plot_outliers(df_outlier: pd.DataFrame, path: str):
    """Save outlier plot from output dataframe of detect_outliers function.

    Args:
        df_outliers (pd.DataFrame): output data of detect_outliers
        path (string): path and name to save plot

    Returns:
        fig (Figure): plot of outlier and inlier from data
    """
    outlier_plot = sns.histplot(x="scores", data=df_outlier, hue="anomaly")
    fig = outlier_plot.get_figure()
    fig.savefig(path)
    return fig
