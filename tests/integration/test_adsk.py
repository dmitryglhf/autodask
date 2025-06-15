from autodask.main import AutoDask


def test_basic_clf_usage(classification_data):
    X, y = classification_data
    adsk = AutoDask(task='classification')
    adsk.fit(X, y)
