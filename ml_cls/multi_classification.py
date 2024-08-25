import numpy as np
import pandas as pd
import os
from ml_cls.ml_base import MLBase
from hydra.utils import instantiate
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from ml_cls.metrics.metrics import Metrics


class MultiClf(MLBase):
    def __init__(self, config):
        self.config = config

    def calculate_metric(self, test, datasets):
        result = {}
        test = test[test['dataset'].isin(datasets)]
        result.update({'balanced_accuracy': round(balanced_accuracy_score(test['class'], test['pred']) * 100, 2)})
        result.update({'accuracy': round(accuracy_score(test['class'], test['pred']) * 100, 2)})
        for average in self.config['ml']['F1']:
            result.update({'f1_' + average: round(f1_score(test['class'], test['pred'], average=average), 2)})
        # conf = {'target_names':[0,1,2,3]}
        # cr = Metrics(**conf)
        # print(cr.get_metrics(test))
        return result
