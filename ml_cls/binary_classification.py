import numpy as np
import pandas as pd
import os
from ml_cls.ml_base import MLBase
from hydra.utils import instantiate
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from ml_cls.metrics.metrics import Metrics

class BinaryClf(MLBase):
    def __init__(self, config):
        self.config = config

    def calculate_metric(self, test, datasets):
        result = {}
        test = test[test['dataset'].isin(datasets)]
        result.update({'balanced_accuracy': round(balanced_accuracy_score(test['class'], test['pred']) * 100, 2)})
        result.update({'accuracy': round(accuracy_score(test['class'], test['pred']) * 100, 2)})
        #tn, fp, fn, tp = confusion_matrix(test['class'], test['pred']).ravel()
        #result.update({'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})
        for average in ['binary', 'macro', 'micro', 'weighted']:
            result.update({'f1_' + average: round(f1_score(test['class'], test['pred'], average=average), 2)})
        return result


