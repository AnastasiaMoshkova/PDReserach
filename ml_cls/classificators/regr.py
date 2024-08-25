from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from abc import ABC
class BaseRegr(ABC):
    def __init__(self, **config):
        self.config = config
        self.regr = None

    def fit(self, x, y):
        self.regr = self.regr.fit(x, y)
        return self.regr

    def predict(self, x):
        return self.regr.predict(x)

    def predict_proba(self, x):
        return self.regr.predict_proba(x)

    def predict_from_pretrained(self, path):
        pass
    def score(self, x, y):
        return self.regr.score(x, y)



class DT(BaseRegr):
    def __init__(self, **config):
        super().__init__(**config)
        self.regr = DecisionTreeRegressor(**config)

class RF(BaseRegr):
    def __init__(self, **config):
        super().__init__(**config)
        self.regr = RandomForestRegressor(**config)

class XGBoost(BaseRegr):
    def __init__(self, **config):
        super().__init__(**config)
        self.regr = make_pipeline(StandardScaler(), XGBRegressor(**config))

class SVR(BaseRegr):
    def __init__(self, **config):
        super().__init__(**config)
        self.regr = make_pipeline(StandardScaler(), svm.SVR(**config))

class KNN(BaseRegr):
    def __init__(self, **config):
        super().__init__(**config)
        self.regr = make_pipeline(StandardScaler(), KNeighborsRegressor(**config))



