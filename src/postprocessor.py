from collections import Counter
import numpy as np

class BruteForce:

    def __init__(self, clf, X, y, A, preprocessor):
        # clf: the learned classifier
        # X: independent variables (2-d pd.DataFrame)
        # y: the dependent variable (1-d np.array, binary y in {0,1})
        # A: the name of the sensitive attribute with binary values (string)
        # preprocessor: the data preprocessor for clf
        self.clf = clf
        self.A = A
        self.preprocessor = preprocessor
        self.pos_label = list(self.clf.classes_).index(1)
        groups = {}
        for group in X[A].unique():
            groups[group] = X[X[self.A] == group].index.to_list()
        self.fit(self.clf.predict_proba(self.preprocessor.transform(X))[:, self.pos_label], y, groups)

    def predict(self, X):
        # X: independent variables (2-d pd.DataFrame)
        # Return: the final class prediction for one data point belonging to the specific sensitive group
        pred_proba = self.clf.predict_proba(self.preprocessor.transform(X))[:, self.pos_label]
        group = X[self.A]
        pred = np.array([0 if pred_proba[i] < self.thresholds[A1] else 1 for i, A1 in enumerate(group)])
        return pred

    def fit(self, pred_proba, y, groups):
        # pred_proba: 1-d np.array, prediction probability output from a classifier
        # y: 1-d np.array, actual labels
        # groups: dictionary, {Ai: [indices of training data points from sensitive group Ai]}
        # self.thresholds: dictionary, {Ai: threshold for group Ai}
        self.thresholds = {}
        for key in groups:
            order = np.argsort(pred_proba[groups[key]])
            error = Counter(y[groups[key]])[0]
            errors = [error]
            for i, label in enumerate(y[np.array(groups[key])[order]]):
                if label > 0:
                    error+=1
                else:
                    error-=1
                errors.append(error)
            id = np.argmin(errors)
            if id == len(order):
                thres = 1.0
            else:
                thres = pred_proba[groups[key]][order[id]]
            self.thresholds[key] = thres




