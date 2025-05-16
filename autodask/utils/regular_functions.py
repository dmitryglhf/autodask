import numpy as np

from autodask.repository.metric_repository import get_metric, get_default_metric


def is_classification_task(task: str) -> bool:
    if task == "classification":
        return True
    elif task == "regression":
        return False
    else:
        raise ValueError(f"Unsupported task: {task}")


def setup_metric(metric_name:str=None, task:str=None):
    """Set up the metric function based on task or provided metric name"""
    if task is None:
        raise ValueError("There is no task to get metrics")
    if metric_name is None:
        score_func, metric_name, maximize_metric = get_default_metric(task)
    score_func, metric_name, maximize_metric =  get_metric(metric_name)
    return score_func, metric_name, maximize_metric


def get_n_classes(y):
    return len(np.unique(y))
