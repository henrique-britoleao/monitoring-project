import sys

sys.path.insert(0, "Loading/")

import pandas as pd
import json
import os

import constants as cst
import loading
import train_model


def train_drift_adjusted_model(batch_id: int) -> pd.DataFrame:
    drift_type = check_drift_type(batch_id)

    batch_name = cst.BATCH_NAME_TEMPLATE.substitute(id=batch_id)
    batch_df = loading.read_csv_from_path(os.path.join(cst.BATCHES_PATH, batch_name))

    sample_df = loading.load_training_data()

    if drift_type == "Covariate drift":
        new_training_data = pd.concat([sample_df, batch_df])
    elif drift_type == "Concept drift":
        new_training_data = batch_df

    train_model.train_model_pipeline(
        data=new_training_data,
        predictions_path=cst.PREDICTED_TRAIN_FILE_PATH,
        model_name=f"drift_adjusted_{cst.selected_dataset}_{cst.selected_model}",
    )


def check_drift_type(batch_id: int) -> str:
    with open(cst.MONITORING_METRICS_FILE_PATH, "r") as config:
        dict_metrics = json.load(config)

    numerical_metrics = (
        pd.DataFrame(
            dict_metrics[0]["batch{}.csv".format(batch_id)]["metrics"][
                "covariate_drift_metrics"
            ]["numerical_metrics"]
        )
        .applymap(lambda x: dict(x)["alert"])
        .transpose()
    )
    categorical_metrics = (
        pd.DataFrame(
            dict_metrics[0]["batch{}.csv".format(batch_id)]["metrics"][
                "covariate_drift_metrics"
            ]["categorical_metrics"]
        )
        .applymap(lambda x: dict(x)["alert"])
        .transpose()
    )
    binary_metrics = (
        pd.DataFrame(
            dict_metrics[0]["batch{}.csv".format(batch_id)]["metrics"][
                "covariate_drift_metrics"
            ]["binary_metrics"]
        )
        .applymap(lambda x: dict(x)["alert"])
        .transpose()
    )

    covariate_drift = (
        numerical_metrics.append(categorical_metrics).append(binary_metrics).fillna(0)
    )
    concept_drift = dict_metrics[0]["batch{}.csv".format(batch_id)]["metrics"]["PSI"][
        "alert"
    ]

    if covariate_drift.sum().sum() == 0 and concept_drift == 1:
        drift_type = "Concept drift"
    elif covariate_drift.sum().sum() == 0 and concept_drift == 0:
        drift_type = "No shift"
    else:
        drift_type = "Covariate drift"

    return drift_type


if __name__ == "__main__":
    train_drift_adjusted_model(1)
