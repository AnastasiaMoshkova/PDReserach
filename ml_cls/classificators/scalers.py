from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class ScalerCustom():
    def __init__(self, **config):
        self.scaler = None

    def fit(self, X, y = None):
        return self.scaler.fit(X,y)

    def fit_transform(self, X, y = None):
        return self.scaler.fit_transform(X,y)

    def transform(self, X, copy=None):
        return self.scaler.transform(X,copy)

    def inverse_transform(self, X, copy=None):
        return self.scaler.inverse_transform(X, copy)

class StandardScalerCustom(ScalerCustom):
    def __init__(self, **config):
        self.scaler = StandardScaler(**config)

class MinMaxScalerCustom(ScalerCustom):
    def __init__(self, **config):
        self.scaler = MinMaxScaler(**config)

class RobustScalerCustom(ScalerCustom):
    def __init__(self, **config):
        self.scaler = RobustScaler(**config)

