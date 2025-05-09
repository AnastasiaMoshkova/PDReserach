from supervised.automl import AutoML
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from ml_cls.ml_base import MLBase


class AutoMLJar(MLBase):

    def __init__(self, config):
        self.config = config

    def processing(self, output_dir):
        features = self.config['features']
        df = self.dataset_correction(features)
        df = df[df['dataset'].isin[self.config['datasets']]]
        df['class'] = df['stage'].apply(int)

        if self.config['type'] == 'loo':
            skgf = LeaveOneGroupOut() #TODO
        if self.config['type'] == 'loo':
            skgf = StratifiedGroupKFold(n_splits=8)
        folds_array = list(skgf.split(df[features], df['class'], groups=df['id']))
        validation_strategy = {"validation_type": "custom", 'folds': folds_array}

        automl = AutoML(algorithms=self.config['algorithms'],
                        mode=self.config['mode'],
                        ml_task=self.config['ml_task'],
                        eval_metric=self.config['eval_metric'],
                        results_path=output_dir,
                        total_time_limit=60 * 60 * self.config['total_time_limit_hours'],
                        explain_level=self.config['explain_level'],
                        validation_strategy=validation_strategy)

        df['sample_weight'] = compute_sample_weight(class_weight="balanced", y=df['class'])
        automl.fit(df[features], df['class'].values, sample_weight=df['sample_weight'].values, cv=folds_array)