import numpy as np
import pandas as pd
import os
from ml_cls.ml_base import MLBase
from hydra.utils import instantiate
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

class Regression(MLBase):
    def __init__(self, config):
        self.config = config

    def calculate_metric(self, test, datasets):
        result = {}
        test = test[test['dataset'].isin(datasets)]
        #print(test[test['regr']==0][['regr','pred']])
        #test.loc[test['pred']<0,'pred'] = 0
        #test['pred'] = abs(test['pred'])
        result.update({'r2': round(r2_score(test['regr'], test['pred']), 3)})
        result.update({'rmse': round(np.sqrt(mean_squared_error(test['regr'], test['pred'])), 3)})
        result.update({'mae': round(mean_absolute_error(test['regr'], test['pred']), 3)})
        return result

    def calculate_info_metric(self, test):
        result = {}
        for dataset in self.config['ml']['datasets']:
            res = test[test['dataset'] == dataset]
            result.update({dataset: round(accuracy_score(res['class'], round(res['pred'])) * 100, 2), 'data_' + dataset: int(res.shape[0])})
        for stage in [0, 1, 2, 2.5, 3, 3.5]:
            res = test[test['stage'] == stage]
            result.update({'stage' + str(stage): round(accuracy_score(res['class'], round(res['pred'])) * 100, 2), 'data_stage' + str(stage): int(res.shape[0])})
        for class_type in [0, 1, 2, 3]:
            res = test[test['class'] == class_type]
            res['pred'] = round(res['pred'])
            result.update({'class' + str(class_type): round(accuracy_score(res['class'], round(res['pred'])) * 100, 2),'data_class' + str(class_type): int(res.shape[0])})
        return result

