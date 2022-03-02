import numpy as np
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from scipy import sparse

class Optimizer:

    def __init__(self, lam = 1000.0, alpha = 0.1, max_iter = 100, tol = 0.00001):
        # lam: weight to trade-off between error and bias
        # alpha: learning rate
        # max_iter: maximum iteration until stop
        # tol: early stopping rule if max{|gradient_i|, i = 1, ..., n} <= tol
        self.lam = lam
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def score(self, X):
        # Calculate the prediction score for the input data X
        # score = sigmoid(w*[x,1]) for each x in X
        XX = sparse.hstack((X, np.array([1] * X.shape[0])[:, None]))
        linear = XX*self.w
        return 1/(1+np.exp(-linear))

    def predict(self, X):
        scores = self.predict_proba(X)
        return np.array([1 if score >= 0.5 else 0 for score in scores])

    def predict_proba(self, X):
        X_test = sparse.csr_matrix(self.preprocessor.transform(X))
        return self.score(X_test)

    def gradient(self, X, y, groups):
        # Calculate gradient

        def C_gradient(Xsub):
            XX = sparse.hstack((Xsub, np.array([1]*Xsub.shape[0])[:, None]))
            linear = XX*self.w
            return np.exp(-linear) / ((1 + np.exp(-linear)) ** 2)*XX

        n = len(y)
        C_grad = C_gradient(X)
        error_grad = 2/(n**2)*np.sum(self.score(X)-y)*C_grad
        C_grad0 = C_gradient(X[groups[0]])
        C_grad1 = C_gradient(X[groups[1]])
        bias_grad = 2 * (np.sum(self.score(X[groups[0]]) - y[groups[0]]) / len(groups[0]) - np.sum(
            self.score(X[groups[1]]) - y[groups[1]]) / len(groups[1])) * (
                                C_grad0 / len(groups[0]) - C_grad1 / len(groups[1]))
        return error_grad+bias_grad*self.lam

    def loss(self, X, y, groups):
        return (np.sum(self.score(X) - y) / len(y)) ** 2 + self.lam * (
                    np.sum(self.score(X[groups[0]]) - y[groups[0]]) / len(groups[0]) - np.sum(
                self.score(X[groups[1]]) - y[groups[1]]) / len(groups[1]))**2

    def fit(self, X, y, A):
        # X: independent variables (2-d pd.DataFrame)
        # y: the dependent variable (1-d np.array, binary y in {0,1})
        # A: the name of the sensitive attribute with binary values (string)

        # Preprocess training data
        numerical_columns_selector = selector(dtype_exclude=object)
        categorical_columns_selector = selector(dtype_include=object)

        numerical_columns = numerical_columns_selector(X)
        categorical_columns = categorical_columns_selector(X)

        categorical_preprocessor = OneHotEncoder()
        numerical_preprocessor = StandardScaler()
        self.preprocessor = ColumnTransformer([
            ('OneHotEncoder', categorical_preprocessor, categorical_columns),
            ('StandardScaler', numerical_preprocessor, numerical_columns)])

        X_train = sparse.csr_matrix(self.preprocessor.fit_transform(X))
        clf_init = LogisticRegression(max_iter=1000)
        clf_init.fit(X_train, y)
        self.w = np.array(list(clf_init.coef_[0])+list(clf_init.intercept_))

        # Obtain indices of data belonging to each group
        groups = []
        for group in X[A].unique():
            sub = X[X[A] == group].index.to_list()
            groups.append(sub)
        for iter in range(self.max_iter):
            grad = self.gradient(X_train, y, groups)
            if max(np.abs(grad)) <= self.tol:
                break
            self.w -= self.alpha * grad
            # print(self.loss(X_train, y, groups))







