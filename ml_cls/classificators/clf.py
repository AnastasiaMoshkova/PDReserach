from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import svm
from abc import ABC
class BaseClf(ABC):
    def __init__(self, **config):
        self.config = config
        self.cls = None

    def fit(self, x, y):
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

class DT(BaseClf):
    def __init__(self, **config):
        super().__init__(**config)
        self.cls = DecisionTreeClassifier(**config)

class RF(BaseClf):
    def __init__(self, **config):
        super().__init__(**config)
        self.cls = RandomForestClassifier(**config)

class XGBoost(BaseClf):
    def __init__(self, **config):
        super().__init__(**config)
        self.cls = XGBClassifier(**config)

class SVM(BaseClf):
    def __init__(self, **config):
        super().__init__(**config)
        self.cls = svm.SVC(**config)

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

class LGBM(BaseClf):
    def __init__(self, **config):
        super().__init__(**config)
        self.cls = lgb.LGBMClassifier(**config)


