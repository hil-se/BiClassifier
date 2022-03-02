from collections import Counter

class Metrics:

    def __init__(self, clf, X_test, y_test, preprocessor=None):
        # clf: the classifier/predictor to be evaluated
        # X_test: test data, independent variables (2-d pd.DataFrame)
        # y_test: test data, the dependent variable (1-d np.array, binary y_test in {0,1})
        # preprocessor: the preprocessor for X_test
        self.clf = clf
        self.X = X_test
        self.y = y_test
        self.preprocessor = preprocessor
        if self.preprocessor:
            self.y_pred = self.clf.predict(self.preprocessor.transform(self.X))
        else:
            self.y_pred = self.clf.predict(self.X)

    def accuracy(self):
        return Counter(self.y==self.y_pred)[True] / len(self.y)

    def tpr(self):
        conf = self.confusion()
        tpr = conf['tp'] / (conf['tp'] + conf['fn'])
        return tpr

    def fpr(self):
        conf = self.confusion()
        fpr = conf['fp'] / (conf['fp'] + conf['tn'])
        return fpr

    def pr(self):
        conf = self.confusion()
        fpr = (conf['fp']+conf['tp']) / (conf['fp'] + conf['tn'] + conf['tp'] + conf['fn'])
        return fpr


    def eod(self, A):
        # A: string, name of the target binary sensitive attribute
        # return: EOD = |TPR(A=A1)-TPR(A=A0)|
        # TPR = #(y=1, C=1) / #(y=1)
        tpr = []
        for group in self.X[A].unique():
            sub = self.X[self.X[A]==group].index.to_list()
            conf = self.confusion(sub)
            tpr.append(conf['tp'] / (conf['tp']+conf['fn']))
        return max(tpr)-min(tpr)

    def aod(self, A):
        # A: string, name of the target binary sensitive attribute A
        # return: AOD = 0.5*|TPR(A=A1)-TPR(A=A0)+FPR(A=A1)-FPR(A=A0)|
        # FPR = #(y=0, C=1) / #(y=0)
        aod = []
        for group in self.X[A].unique():
            sub = self.X[self.X[A] == group].index.to_list()
            conf = self.confusion(sub)
            tpr = conf['tp'] / (conf['tp'] + conf['fn'])
            fpr = conf['fp'] / (conf['fp'] + conf['tn'])
            aod.append((tpr+fpr)/2)
        return max(aod) - min(aod)

    def spd(self, A):
        # A: string, name of the target binary sensitive attribute A
        # return: SPD = |PR(A=A1)-PR(A=A0)|
        # PR = #(C=1) / #X
        pr = []
        for group in self.X[A].unique():
            sub = self.X[self.X[A] == group].index.to_list()
            pr.append(Counter(self.y_pred[sub])[1]/len(sub))
        return max(pr) - min(pr)

    def di(self, A):
        # A: string, name of the target binary sensitive attribute A
        # return: DI = #(C(x, A=A1)!=C(x, A=A0)) / #X
        X_pair = self.X.copy()
        values = self.X[A].unique()
        pair = lambda x: values[1-list(values).index(x)]
        X_pair[A] = X_pair[A].apply(pair)

        if self.preprocessor:
            ypair_pred = self.clf.predict(self.preprocessor.transform(X_pair))
        else:
            ypair_pred = self.clf.predict(X_pair)

        return Counter(ypair_pred!=self.y_pred)[True] / len(self.y_pred)

    def confusion(self, sub=None):
        if sub==None:
            sub = range(len(self.y))
        y = self.y[sub]
        y_pred = self.y_pred[sub]
        conf = {'tp':0, 'tn':0, 'fp':0, 'fn':0}
        for i in range(len(y)):
            if y[i]==0 and y_pred[i]==0:
                conf['tn']+=1
            elif y[i]==1 and y_pred[i]==1:
                conf['tp'] += 1
            elif y[i]==0 and y_pred[i]==1:
                conf['fp'] += 1
            elif y[i]==1 and y_pred[i]==0:
                conf['fn'] += 1
        return conf







