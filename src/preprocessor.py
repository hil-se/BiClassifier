from collections import Counter
import numpy as np
import pandas as pd


def AdversarialTraining(X, y, A):
    # X: independent variables (2-d pd.DataFrame)
    # y: the dependent variable (1-d np.array, binary y in {0,1})
    # A: the name of the sensitive attribute with binary values (string)
    # Return: (X', y') of double size of (X, y). (X', y') include both (X, y, A=A1) and (X, y, A=A2)
    X_pair = X.copy()
    values = X[A].unique()
    pair = lambda x: values[1 - list(values).index(x)]
    X_pair[A] = X_pair[A].apply(pair)
    yout = np.array(list(y) * 2)
    Xout = pd.concat([X, X_pair])
    return Xout, yout

def Reweighing(X, y, A):
    # X: independent variables (2-d pd.DataFrame)
    # y: the dependent variable (1-d np.array, binary y in {0,1})
    # A: the name of the sensitive attribute with binary values (string)
    grouping = {}
    for i, label in enumerate(y):
        key = (X[A][i], label)
        if key not in grouping:
            grouping[key]=[]
        grouping[key].append(i)
    class_weight = Counter(y)
    group_weight = Counter(X[A])
    sample_weight = np.array([1.0]*len(y))
    for key in grouping:
        weight = class_weight[key[-1]]*group_weight[key[0]]/len(grouping[key])
        for i in grouping[key]:
            sample_weight[i] = weight
    # Rescale the total weights to len(y)
    sample_weight = sample_weight * len(y) / sum(sample_weight)
    return sample_weight

def FairBalance(X, y, A, class_balance = False):
    # X: independent variables (2-d pd.DataFrame)
    # y: the dependent variable (1-d np.array, binary y in {0,1})
    # A: the name of the sensitive attribute with binary values (string)
    # class_balance: whether to balance class distribution. True: FairBalanceClass, False: FairBalance
    grouping = {}
    for i, label in enumerate(y):
        key = (X[A][i], label)
        if key not in grouping:
            grouping[key]=[]
        grouping[key].append(i)
    class_weight = Counter(y)
    if class_balance:
        class_weight = {key: 1.0 for key in class_weight}
    sample_weight = np.array([1.0]*len(y))
    for key in grouping:
        weight = class_weight[key[-1]]/len(grouping[key])
        for i in grouping[key]:
            sample_weight[i] = weight
    # Rescale the total weights to len(y)
    sample_weight = sample_weight * len(y) / sum(sample_weight)
    return sample_weight


def ClassBalance(X, y, A):
    # X: independent variables (2-d pd.DataFrame)
    # y: the dependent variable (1-d np.array, binary y in {0,1})
    # A: the name of the sensitive attribute with binary values (string)
    grouping = {}
    for i, label in enumerate(y):
        key = (X[A][i], label)
        if key not in grouping:
            grouping[key]=[]
        grouping[key].append(i)
    group_weight = Counter(X[A])
    sample_weight = np.array([1.0]*len(y))
    for key in grouping:
        weight = group_weight[key[0]]/len(grouping[key])
        for i in grouping[key]:
            sample_weight[i] = weight
    # Rescale the total weights to len(y)
    sample_weight = sample_weight * len(y) / sum(sample_weight)
    return sample_weight
