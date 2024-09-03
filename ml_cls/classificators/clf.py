from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import svm
from abc import ABC
import numpy as np

class BaseClf(ABC):
    def __init__(self, **config):
        self.config = config
        self.cls = None

    def fit(self, x, y, sample_weight=None):
        self.cls = self.cls.fit(x, y)
        return self.cls

    def predict(self, x):
        return self.cls.predict(x)

    def predict_proba(self, x):
        return self.cls.predict_proba(x)

    def predict_from_pretrained(self, path):
        pass


class NB(BaseClf):
    def __init__(self, **config):
        super().__init__(**config)
        self.cls = GaussianNB(**config) #make_pipeline(StandardScaler(), GaussianNB()) #GaussianNB()
    def fit(self, x, y, sample_weight=None):
        self.cls = self.cls.fit(x, y, sample_weight=sample_weight)
        return self.cls

class DT(BaseClf):
    def __init__(self, **config):
        super().__init__(**config)
        self.cls = DecisionTreeClassifier(**config)
    def fit(self, x, y, sample_weight=None):
        self.cls = self.cls.fit(x, y, sample_weight=sample_weight)
        return self.cls

class RF(BaseClf):
    def __init__(self, **config):
        super().__init__(**config)
        self.cls = RandomForestClassifier(**config)
    def fit(self, x, y, sample_weight=None):
        self.cls = self.cls.fit(x, y, sample_weight=sample_weight)
        return self.cls

class XGBoost(BaseClf):
    def __init__(self, **config):
        super().__init__(**config)
        self.cls = XGBClassifier(**config)

    def fit(self, x, y, sample_weight=None):
        self.cls = self.cls.fit(x, y, sample_weight=sample_weight)
        return self.cls

class SVM(BaseClf):
    def __init__(self, **config):
        super().__init__(**config)
        self.cls = svm.SVC(**config)
    def fit(self, x, y, sample_weight=None):
        self.cls = self.cls.fit(x, y, sample_weight=sample_weight)
        return self.cls

class KNN(BaseClf):
    def __init__(self, **config):
        super().__init__(**config)
        self.cls = KNeighborsClassifier(**config)

class LDA(BaseClf):
    def __init__(self, **config):
        super().__init__(**config)
        self.cls = LinearDiscriminantAnalysis(**config)


class LR(BaseClf):
    def __init__(self, **config):
        super().__init__(**config)
        self.cls = LogisticRegression(**config)
    def fit(self, x, y, sample_weight=None):
        self.cls = self.cls.fit(x, y, sample_weight=sample_weight)
        return self.cls

class LGBM(BaseClf):
    def __init__(self, **config):
        super().__init__(**config)
        self.cls = lgb.LGBMClassifier(**config)
    def fit(self, x, y, sample_weight=None):
        self.cls = self.cls.fit(x, y, sample_weight=sample_weight)
        return self.cls

class ExtraTree(BaseClf):
    def __init__(self, **config):
        super().__init__(**config)
        self.cls = ExtraTreesClassifier(**config)
    def fit(self, x, y, sample_weight=None):
        self.cls = self.cls.fit(x, y, sample_weight=sample_weight)
        return self.cls


