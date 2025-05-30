from sklearn import metrics
from typing import Callable, Dict, Any, Tuple
import numpy as np


# Main metrics repository with their meta-data
METRICS_REPOSITORY: Dict[str, Dict[str, Any]] = {
    # Classification metrics
    'accuracy': {
        'func': metrics.accuracy_score,
        'task': 'classification',
        'maximize': True,
    },
    'f1': {
        'func': metrics.f1_score,
        'task': 'classification',
        'maximize': True,
    },
    'precision': {
        'func': metrics.precision_score,
        'task': 'classification',
        'maximize': True,
    },
    'recall': {
        'func': metrics.recall_score,
        'task': 'classification',
        'maximize': True,
    },
    'roc_auc': {
        'func': metrics.roc_auc_score,
        'task': 'classification',
        'maximize': True,
    },
    'log_loss': {
        'func': metrics.log_loss,
        'task': 'classification',
        'maximize': False,
        'default_for_task': True
    },

    # Regression metrics
    'mse': {
        'func': metrics.mean_squared_error,
        'task': 'regression',
        'maximize': False,
        'default_for_task': True
    },
    'r2': {
        'func': metrics.r2_score,
        'task': 'regression',
        'maximize': True
    },
    'mae': {
        'func': metrics.mean_absolute_error,
        'task': 'regression',
        'maximize': False
    },
    'rmse': {
        'func': lambda y_true, y_pred: np.sqrt(metrics.mean_squared_error(y_true, y_pred)),
        'task': 'regression',
        'maximize': False
    }
}


def get_metric(metric_name: str, task: str = None) -> Tuple[Callable, str, bool]:
    """Return metric and their properties"""
    if metric_name not in METRICS_REPOSITORY:
        raise ValueError(f"Metric '{metric_name}' not found in repository. "
                         f"Available metrics: {list(METRICS_REPOSITORY.keys())}")

    metric_info = METRICS_REPOSITORY[metric_name]

    if task and metric_info['task'] != task:
        raise ValueError(f"Metric '{metric_name}' is for {metric_info['task']}, "
                         f"but current task is {task}")

    return metric_info['func'], metric_name, metric_info['maximize']


def get_default_metric(task: str) -> Tuple[Callable, str, bool]:
    """Return default metric"""
    for metric_name, metric_info in METRICS_REPOSITORY.items():
        if metric_info.get('task') == task and metric_info.get('default_for_task'):
            return metric_info['func'], metric_name, metric_info['maximize']
    raise ValueError(f"No default metric found for task: {task}")
