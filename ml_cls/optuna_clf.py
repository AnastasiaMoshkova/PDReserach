import pandas as pd
import os
import optuna
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GroupKFold, LeaveOneGroupOut, cross_val_predict
from ml_cls.ml_base import MLBase
from hydra.utils import instantiate
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import r2_score
import sklearn
from sklearn.utils.class_weight import compute_sample_weight,compute_class_weight



class OptunaClf(MLBase):
    def __init__(self, config):
        self.config = config
        self.X, self.y,self.group  = self.dataset_processing()
        print(self.group.shape)

    def dataset_processing(self):
        dataset = self.dataset_correction(self.config['ml']['features']).dropna(subset=self.config['ml']['features'])
        dataset = self.data_processing_cv(dataset)
        dataset = dataset[dataset['dataset'].isin(self.config['ml']['datasets'])]
        #dataset = dataset.groupby(['id', 'r']).mean(numeric_only=True)
        X, y = dataset[self.config['ml']['features']], dataset['class']
        print(X.shape, y.shape)
        return X,y,dataset['group']

    def objective(self, trial, clf):
        scalers = trial.suggest_categorical("scalers", self.config['ml']['scalers'])

        # TODO from class scaler
        if scalers == "minmax":
            scaler = MinMaxScaler()
        elif scalers == "standard":
            scaler = StandardScaler()
        elif scalers == "robust":
            scaler = RobustScaler()
        else:
            scaler = None

        dim_red = trial.suggest_categorical("dim_red", self.config['ml']['dim_red'])

        # TODO from class dimentional_reducion
        if dim_red == "PCA":
            pca_n_components=trial.suggest_int("pca_n_components", 2, 5) # suggest an integer from 2 to 30
            dimen_red_algorithm=PCA(n_components=pca_n_components)
        else:
            dimen_red_algorithm='passthrough'

        sample_weight = trial.suggest_categorical("sample_weight", self.config['ml']['sample_weight'])

        params = {'_target_':self.config['ml'][clf]['_target_']}
        if 'suggest_int' in self.config['ml'][clf].keys():
            for key in self.config['ml'][clf]['suggest_int'].keys():
                parameters = self.config['ml'][clf]['suggest_int'][key]
                params[key] = trial.suggest_int(key, parameters['first'], parameters['last'], parameters['step'])
        if 'suggest_categorical' in self.config['ml'][clf].keys():
            for key in self.config['ml'][clf]['suggest_categorical'].keys():
                params[key] = trial.suggest_categorical(key, self.config['ml'][clf]['suggest_categorical'][key])
        if 'suggest_float' in self.config['ml'][clf].keys():
            for key in self.config['ml'][clf]['suggest_float'].keys():
                params[key] = trial.suggest_float(key, self.config['ml'][clf]['suggest_float'][key])

        estimator = instantiate(params)

        # -- Make a pipeline
        #pipeline = make_pipeline(scaler, dimen_red_algorithm, estimator)
        pipeline = sklearn.pipeline.Pipeline(
            [('scaler', scaler),
            ('dim_red', dimen_red_algorithm),
            ('estimator',estimator)]) #TODO
        #pipeline.set_params({'estimator__sample_weight' : self.y})

        #TODO from class mlbase
        #print(cv.get_n_splits(self.X, self.y, groups = self.group,))
        if self.config['ml']['cv']['type']=='cv_loo':
            cv = LeaveOneGroupOut()
            if sample_weight:
                weights = compute_sample_weight(class_weight="balanced", y=self.y)
                predict = cross_val_predict(pipeline, self.X, self.y, groups = self.group, cv = cv, params = {'estimator__sample_weight' : weights}) #, cross_val_score scoring=self.config['ml']['scoring']
            else:
                predict = cross_val_predict(pipeline, self.X, self.y, groups=self.group, cv=cv)
            if self.config['ml']['scoring']=='balanced_accuracy':
                metric = balanced_accuracy_score(self.y, predict)
            if self.config['ml']['scoring']=='r2':
                metric = r2_score(self.y, predict)
        if self.config['ml']['cv']['type']=='cv_k_folds':
            cv = GroupKFold(n_splits=self.config['ml']['cv']['folds'])
            score = cross_val_score(pipeline, self.X, self.y, groups=self.group, cv=cv,scoring=self.config['ml']['scoring'])
            metric = score.mean() # calculate the mean of scores
        return metric

    def processing(self, output_dir):
        res = []
        for clf in self.config['ml']['classifiers']:
            study = optuna.create_study(direction="maximize") # maximise the score during tuning
            study.optimize(lambda trial: self.objective(trial, clf), n_trials=self.config['ml']['trials']) # run the objective function 100 times
            print(study.best_trial)  # print the best performing pipeline
            res.append(
                {'clf': clf,
                 'value': study.best_trial.value,
                 'metric': self.config['ml']['scoring'],
                 'cv':'LeaveOneGroupOut',
                 'scaler':study.best_trial.params['scalers'],
                 'dim_red':study.best_trial.params['dim_red'],
                 'parameters': study.best_trial.params,
                 'n_trials':self.config['ml']['trials'],
                 'dataset_folder':self.config['folder'],
                 'features': self.config['ml']['features']})
        df = pd.DataFrame(res)
        df.to_csv(os.path.join(output_dir, 'optuna_results.csv'))