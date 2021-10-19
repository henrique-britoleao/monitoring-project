import json

def save_training_performance_metrics(metrics: dict, conf: dict) -> None:
    """
    Saves the dictionary containing model performance metrics to a json file

    Args:
        metrics (dict): Dict of classification performance metrics
        conf (dict): Configuration file stored as a json object
    """
    with open(conf['paths']['Outputs_path'] + conf['paths']['folder_metrics'] + 'training_metrics_'
            + conf['selected_dataset'] + "_" + conf['selected_model'] + '.txt', 'w') as outfile:
        json.dump(str(metrics), outfile)