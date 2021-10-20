import json

def alert(
    value: float,
    metric: str,
    data_type: str,
    drift_type: str,
    sign,
):
    """outputs alert 0 or 1 depending on treshhold rule
    a < b -> lt(a, b)
    a <= b -> le(a, b)
    a >= b -> ge(a, b)
    a > b -> gt(a, b)
    Args:
        value (int) : value of metric
        metric (str): metric we want to evalue
        sign ([type], optional): bigger than, smaller than. Defaults to ge.
    """
    path = "../params/conf/conf.json"

    with open(path, 'r') as config:
        dict_config = json.load(config)

    if drift_type == "concept_drift":
        threshold = dict_config["dict_info_files"]['marketing']["metrics_setup"][drift_type][metric]['threshold']
    else:
        threshold = dict_config["dict_info_files"]['marketing']["metrics_setup"][drift_type][data_type][metric]['threshold']

    if sign(value, threshold):
        alert = 1
    else:
        alert = 0
    return alert