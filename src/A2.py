from metrics import Metrics
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from preprocessor import ClassBalance, FairBalance
from pdb import set_trace


# Implement the Metrics class in metrics.py
# Expected output for executing A1.py:
# Accuracy: 0.852405
# EOD: 0.087291
# AOD: 0.083063
# SPD: 0.177683
# DI: 0.091763

if __name__ == "__main__":
    # data_train_path = "../data/adult_train.csv"
    # data_test_path = "../data/adult_test.csv"
    data_train_path = "../data/compas_train.csv"
    data_test_path = "../data/compas_test.csv"

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

    # Train model
    sample_weight = ClassBalance(X, y, protected)
    # sample_weight = FairBalance(X, y, protected, class_balance=True)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y, sample_weight = sample_weight)

    m_train = Metrics(clf, X, y, preprocessor)
    m_test = Metrics(clf, X_test, y_test, preprocessor)


    print("Training: ")
    print("Accuracy: %f" % m_train.accuracy())
    print("AO: %f" % (m_train.tpr()+m_train.fpr()))
    print("EOD: %f" % m_train.eod(protected))
    print("AOD: %f" % m_train.aod(protected))
    print("SPD: %f" % m_train.spd(protected))

    print("")
    print("Testing: ")
    print("Accuracy: %f" % m_test.accuracy())
    print("AO: %f" % (m_test.tpr() + m_test.fpr()))
    print("EOD: %f" % m_test.eod(protected))
    print("AOD: %f" % m_test.aod(protected))
    print("SPD: %f" % m_test.spd(protected))
