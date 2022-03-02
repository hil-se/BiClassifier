from metrics import Metrics
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from pdb import set_trace


# Implement the Metrics class in metrics.py
# Expected output for executing A1.py:
# Accuracy: 0.852405
# EOD: 0.087291
# AOD: 0.083063
# SPD: 0.177683
# DI: 0.091763

if __name__ == "__main__":
    data_train_path = "../data/adult_train.csv"
    data_test_path = "../data/adult_test.csv"
    # data_train_path = "../data/compas_train.csv"
    # data_test_path = "../data/compas_test.csv"

    #  Load training data
    data_train = pd.read_csv(data_train_path)
    protected = "sex"

    # Separate independent variables and dependent variables
    independent = data_train.keys().tolist()
    dependent = independent.pop(-1)

    X = data_train[independent]
    y = np.array(data_train[dependent])

    # Preprocess
    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)

    numerical_columns = numerical_columns_selector(X)
    categorical_columns = categorical_columns_selector(X)

    categorical_preprocessor = OneHotEncoder()
    numerical_preprocessor = StandardScaler()
    preprocessor = ColumnTransformer([
        ('OneHotEncoder', categorical_preprocessor, categorical_columns),
        ('StandardScaler', numerical_preprocessor, numerical_columns)])

    X_train = preprocessor.fit_transform(X)

    # Load testing data
    data_test = pd.read_csv(data_test_path)

    # Preprocess
    X_test = data_test[independent]
    y_test = np.array(data_test[dependent])

    # Train model for each group
    clfs = {}
    groups = {}
    m_train = {}
    m_test = {}
    for g in X[protected].unique():
        groups[g] = X[X[protected] == g].index.to_list()
        # Fit model
        clf = LogisticRegression(max_iter=1000, class_weight = 'balanced')
        # clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train[groups[g]], y[groups[g]])
        clfs[g] = clf
        # Test on training
        m_train[g] = Metrics(clf, X.loc[groups[g]], y[groups[g]], preprocessor)
        # Test on testing
        test_g = X_test[X_test[protected] == g].index.to_list()
        m_test[g] = Metrics(clf, X_test.loc[test_g], y_test[test_g], preprocessor)

    print("Training: ")
    print("Accuracy Male: %f" % m_train["Male"].accuracy())
    print("Accuracy Female: %f" % m_train["Female"].accuracy())
    print("AO Male: %f" % (m_train["Male"].tpr()+m_train["Male"].fpr()))
    print("AO Female: %f" % (m_train["Female"].tpr()+m_train["Female"].fpr()))
    print("EOD: %f" % (m_train["Male"].tpr() - m_train["Female"].tpr()))
    print("AOD: %f" % (
            (m_train["Male"].tpr() - m_train["Female"].tpr() + m_train["Male"].fpr() - m_train["Female"].fpr()) / 2))
    print("SPD: %f" % (m_train["Male"].pr() - m_train["Female"].pr()))

    print("")
    print("Testing: ")
    print("Accuracy Male: %f" % m_test["Male"].accuracy())
    print("Accuracy Female: %f" % m_test["Female"].accuracy())
    print("AO Male: %f" % (m_test["Male"].tpr() + m_test["Male"].fpr()))
    print("AO Female: %f" % (m_test["Female"].tpr() + m_test["Female"].fpr()))
    print("EOD: %f" % (m_test["Male"].tpr() - m_test["Female"].tpr()))
    print("AOD: %f" % (
            (m_test["Male"].tpr() - m_test["Female"].tpr() + m_test["Male"].fpr() - m_test["Female"].fpr()) / 2))
    print("SPD: %f" % (m_test["Male"].pr() - m_test["Female"].pr()))
