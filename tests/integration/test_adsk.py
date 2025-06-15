from autodask.main import AutoDask


def test_basic_multiclf_usage(classification_data):
    X, y = classification_data
    adsk = AutoDask(task='classification')
    adsk.fit(X, y)

def test_basic_binclf_usage(binary_classification_data):
    X, y = binary_classification_data
    adsk = AutoDask(task='classification')
    adsk.fit(X, y)

def test_basic_reg_usage(regression_data):
    X, y = regression_data
    adsk = AutoDask(task='regression')
    adsk.fit(X, y)
