import numpy as np
import pandas as pd
import os
from hydra.utils import instantiate
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
import pandas as pd
import os
import optuna
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GroupKFold, LeaveOneGroupOut, cross_val_predict
from hydra.utils import instantiate
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import r2_score
from sklearn.impute import KNNImputer, SimpleImputer
from ml_cls.metrics.metrics import confusion_matrix_plot

class MLBase():

    def __init__(self, config):
        self.config = config


    def fill_DI_healthy(self, df, features): #TODO check
        features_id = ['key_id', 'dataset', 'stage', 'id', 'r', 'm', 'hand']
        df['key_id'] = df['id'] + '_' + df['r'].apply(str)
        hc1 = df[(df['dataset'] == 'HEALTHY') & (df['r'] == 0)][['id', 'DI_Anger', 'DI_Disgust', 'DI_Fear','DI_Happiness','DI_Sadness','DI_Surprise']].groupby('id').mean()
        hc2 = df[(df['dataset'] == 'HEALTHY') & (df['r'] != 0)].drop(columns=['DI_Anger', 'DI_Disgust', 'DI_Fear','DI_Happiness','DI_Sadness','DI_Surprise'])
        df_hc = hc2.merge(hc1, left_on='id', right_on='id')
        features_id.extend(features)
        df_hc2 = df[(df['dataset'] == 'HEALTHY') & (df['r'] == 0)][features_id]
        df_pc = df[(df['dataset'] == 'PD')][features_id]
        df_st = df[(df['dataset'] == 'STUDENT')][features_id]
        return pd.concat([df_hc, df_hc2, df_pc, df_st]) #.reset_index(drop=True)

    def fill_nan_by_values_by_user(self): #TODO
        pass

    def fill_nan_by_mean_by_dataset(self): #TODO
        pass

    def fill_nan_by_knn_unsupervised(self, df, features): #TODO
        imputer = KNNImputer(n_neighbors=10, missing_values=np.nan).set_output(transform="pandas")
        #df.to_csv('D:\\PDResearch\\outputs\\df.csv')
        df.loc[:, features] = imputer.fit_transform(df[features])
        #df.to_csv('D:\\PDResearch\\outputs\\df2.csv')
        return df

    def fill_nan_by_knn_supervised(self, df, features): #TODO
        imputer = KNNImputer(n_neighbors=10, missing_values=np.nan).set_output(transform="pandas")
        df.loc[:, features] = imputer.fit_transform(df[features], df['stage'])
        return df

    def fill_nan_by_user_mean(self, df, features): #TODO
        imputer = SimpleImputer(missing_values = np.nan, keep_empty_features=True, strategy='mean', fill_value=np.nan).set_output(transform="pandas")
        for user_id in df['id'].unique():
            df.loc[df['id']==user_id, features] = imputer.fit_transform(df[df['id']==user_id][features])
        return df

    def dataset_correction(self, features):
        df = pd.read_csv(os.path.join(self.config['path'], self.config['folder'], self.config['file']))
        if self.config['fill_DI_healthy']:
            df = self.fill_DI_healthy(df, features)
        for key in self.config['ml']['target_regression']['replace']:
            df['stage'] = df['stage'].replace(key, self.config['ml']['target_regression']['replace'][key])

        #fill missing values in dataset
        if self.config['fill_nan'] == 'fill_nan_by_user_mean':
            df = self.fill_nan_by_user_mean(df, features)
        elif self.config['fill_nan'] == 'fill_nan_by_knn_supervised':
            df = self.fill_nan_by_knn_supervised(df, features)
        elif self.config['fill_nan'] == 'fill_nan_by_knn_unsupervised':
            df = self.fill_nan_by_knn_unsupervised(df, features)

        return df

    def classification_results(self, clf, features, train, test):
        scaler = instantiate(self.config['ml'][clf]['pipeline']['scaler'])
        dim_red = instantiate(self.config['ml'][clf]['pipeline']['dim_red'])
        model = instantiate(self.config['ml'][clf]['config'])
        cl = make_pipeline(scaler, dim_red, model)
        clf = cl.fit(train[features], train[self.config['ml']['target']])
        test['pred'] = clf.predict(test[features])
        #print(clf.score(test[features],test[self.config['ml']['target']]))
        return test

    def data_processing_train_test(self, df):
        train = []
        test = []
        for dataset in self.config['ml']['datasets']:
            for stage in self.config[dataset]['stages']:
                user_id_train = [self.config[dataset]['id_name'] + str(number) for number in self.config[dataset]['stages'][stage]['train']]
                df_train = df[df['id'].isin(user_id_train)]
                df_train['class'] = self.config[dataset]['stages'][stage]['class']
                df_train['regr'] = df_train[self.config['ml']['target_regression']['column']]
                train.append(df_train)
                user_id_test = [self.config[dataset]['id_name'] + str(number) for number in self.config[dataset]['stages'][stage]['test']]
                df_test = df[df['id'].isin(user_id_test)]
                df_test['class'] = self.config[dataset]['stages'][stage]['class']
                df_test['regr'] = df_test[self.config['ml']['target_regression']['column']]
                test.append(df_test)

        data_train = pd.concat(train)
        data_test = pd.concat(test)
        return data_train, data_test

    def cv_group(self, df):
        i = 0
        for idx in df['id'].unique():
            df.loc[df['id'] == idx, 'group'] = i
            i = i + 1
        return df

    def data_processing_cv(self, df):
        dfs_cv = []
        for dataset in self.config['ml']['datasets']:
            for stage in self.config[dataset]['stages']:
                folds = self.config['ml']['cv']['folds']
                for i in range(folds):
                    user_id_cv = [self.config[dataset]['id_name'] + str(number) for number in self.config[dataset]['stages'][stage]['cv'][folds][i]]
                    df_cv = df[df['id'].isin(user_id_cv)]
                    df_cv['class'] = self.config[dataset]['stages'][stage]['class']
                    df_cv['regr'] = df_cv[self.config['ml']['target_regression']['column']]
                    df_cv['cv'] = i
                    dfs_cv.append(df_cv)
        user_id_test = [self.config['STUDENT']['id_name'] + str(number) for number in self.config[dataset]['stages'][0]['test']]
        df_cv = df[df['id'].isin(user_id_test)]
        df_cv['class'] = self.config['STUDENT']['stages'][0]['class']
        df_cv['regr'] = df_cv[self.config['ml']['target_regression']['column']]
        dfs_cv.append(df_cv)
        data_cv = pd.concat(dfs_cv)
        data_cv = self.cv_group(data_cv)
        return data_cv

    def fold_result(self, clf, features, train, test):
        train, test = train.dropna(subset=features), test.dropna(subset=features)
        df_test = self.classification_results(clf, features, train, test)
        metrics = self.calculate_metric(df_test, ['PD', 'HEALTHY'])
        metrics.update(self.calculate_info_metric(df_test))
        metrics.update({'train_data': train.shape[0]})
        return metrics, df_test


    def train_test_clf(self, df, features, clf):
        #dfs_metric = []
        train, test = self.data_processing_train_test(df)
        metrics, df_test = self.fold_result(clf, features, train, test)
        df_metric = pd.DataFrame(metrics, index=[clf])
        df_metric['mode'] = 'train_test'
        return df_metric, df_test

    def train_test(self, df, features, path_to_save):
        dfs_metric = []
        #train, test = self.data_processing_train_test(df)
        for clf in self.config['ml']['classifiers']:
            df_metric, df_test = self.train_test_clf(df, features, clf)
            # df_test.to_csv(os.path.join(path_to_save, clf + '.csv'))
            if self.config['ml']['confusion_matrix']:
                confusion_matrix_plot(df_test, ['PD', 'HEALTHY'], path_to_save, clf + '_train_test')
            dfs_metric.append(df_metric)
        df = pd.concat(dfs_metric)
        df.rename(columns={'Unnamed: 0': 'clf'})
        #df['mode'] = 'train_test'
        return df

    def calculate_metric(self, test, datasets):
        result = {}
        test = test[test['dataset'].isin(datasets)]
        result.update({'accuracy': round(accuracy_score(test['class'], test['pred']) * 100, 2)})
        result.update({'balanced_accuracy': round(balanced_accuracy_score(test['class'], test['pred']) * 100, 2)})
        for average in self.config['ml']['F1']:
            result.update({'f1_' + average: round(f1_score(test['class'], test['pred'], average=average), 2)})
        # conf = {'target_names':[0,1,2,3]}
        # cr = Metrics(**conf)
        # print(cr.get_metrics(test))
        return result

    def cv_by_fix_folds(self, df, features, clf):
        df = self.data_processing_cv(df)
        clf_cv = []
        cv_metrics = []
        for i in range(self.config['ml']['cv']['folds']):
            user_id_split_test = list(df[df['cv'] == i]['id'].values)
            user_id_split_test.extend(df[df['dataset'] == 'STUDENT']['id'].values)
            train = df[~df['id'].isin(user_id_split_test)]
            test = df[df['id'].isin(user_id_split_test)]
            metrics, df_test = self.fold_result(clf, features, train, test)
            cv_metrics.append(pd.DataFrame(metrics, index=['cv' + str(i)]))
        df_cv = pd.concat(cv_metrics)
        df_cv_clf = pd.DataFrame(round(df_cv.mean(), 2)).transpose()
        columns = self.config['ml']['print_console']
        df_cv_clf[[col + '_std' for col in columns]] = pd.DataFrame(round(df_cv[columns].std(), 2)).transpose()
        df_cv_clf.index = [clf]
        df_cv_clf['mode'] = 'cv_fix'
        for col in columns:
            df_cv_clf[col + '±SD'] = df_cv_clf[col].apply(str) + ' ± ' + df_cv_clf[col + '_std'].apply(str)
        clf_cv.append(df_cv_clf)
        return pd.concat(clf_cv)

    def cv_k_folds_random(self, df, features, clf):
        df = self.data_processing_cv(df)
        clf_cv = []
        cv_metrics = []
        cross_val = GroupKFold(n_splits=self.config['ml']['cv']['folds'])
        df_cv = df[df['dataset'].isin(['PD','HEALTHY'])]
        for i, (train_index, test_index) in enumerate(cross_val.split(df_cv[features], df_cv[self.config['ml']['target']], df_cv['group'])):
            user_id_split_test = list(df_cv.iloc[test_index]['id'].values)
            user_id_split_test.extend(df[df['dataset'] == 'STUDENT']['id'].values)
            train = df[~df['id'].isin(user_id_split_test)]
            test = df[df['id'].isin(user_id_split_test)]
            metrics, df_test = self.fold_result(clf, features, train, test)
            cv_metrics.append(pd.DataFrame(metrics, index=['cv' + str(i)]))
        df_cv = pd.concat(cv_metrics)
        df_cv_clf = pd.DataFrame(round(df_cv.mean(), 2)).transpose()
        columns = self.config['ml']['print_console']
        df_cv_clf[[col + '_std' for col in columns]] = pd.DataFrame(round(df_cv[columns].std(), 2)).transpose()
        df_cv_clf.index = [clf]
        df_cv_clf['mode'] = 'cv_k_folds'
        for col in columns:
            df_cv_clf[col + '±SD'] = df_cv_clf[col].apply(str) + ' ± ' + df_cv_clf[col + '_std'].apply(str)
        clf_cv.append(df_cv_clf)
        return pd.concat(clf_cv)


    def cv_loo(self, df, features, clf, path_to_save):
        df = self.data_processing_cv(df)
        df_test_folds = []
        clf_cv = []
        cv_metrics = []
        cross_val = LeaveOneGroupOut()
        df_cv = df[df['dataset'].isin(['PD', 'HEALTHY'])].dropna(subset = features)
        for i, (train_index, test_index) in enumerate(cross_val.split(df_cv[features], df_cv[self.config['ml']['target']], df_cv['group'])):
            user_id_split_test = list(df_cv.iloc[test_index]['id'].values)
            user_id_split_test.extend(df[df['dataset'] == 'STUDENT']['id'].values)
            train = df[~df['id'].isin(user_id_split_test)]
            test = df[df['id'].isin(user_id_split_test)]
            metrics, df_test = self.fold_result(clf, features, train, test)
            cv_metrics.append(pd.DataFrame(metrics, index=['cv' + str(i)]))
            df_test_folds.append(df_test)
        df_test_loo = pd.concat(df_test_folds)
        #df_test_loo.to_csv(os.path.join(path_to_save, 'df_loo.csv'))
        metrics = self.calculate_metric(df_test_loo, ['PD', 'HEALTHY'])
        if self.config['ml']['confusion_matrix']:
            confusion_matrix_plot(df_test_loo, ['PD', 'HEALTHY'], path_to_save, clf)
        df_cv = pd.concat(cv_metrics)
        df_cv_clf = pd.DataFrame(round(df_cv.mean(), 2)).transpose()
        columns = self.config['ml']['print_console']
        df_cv_clf[[col + '_std' for col in columns]] = pd.DataFrame(round(df_cv[columns].std(), 2)).transpose()
        df_cv_clf.index = [clf]
        df_cv_clf['mode'] = 'cv_loo'
        for col in columns:
            df_cv_clf[col + '±SD'] = df_cv_clf[col].apply(str) + ' ± ' + df_cv_clf[col + '_std'].apply(str)
        clf_cv.append(df_cv_clf)
        df_m = pd.DataFrame(metrics, index=[clf])
        df_m.columns = [key+'_loo' for key in metrics.keys()]
        clf_cv.append(df_m)
        return pd.concat(clf_cv, axis=1)
    def cv_result_clf(self, df, features, clf, cv_type, path_to_save):
        '''
        clf_cv = []
        for clf in self.config['ml']['classifiers']:
            if self.config['ml']['cv']['random']:
                clf_cv.append(self.cv_random_split(df, features, clf))
            else:
                clf_cv.append(self.cv_by_fix_folds(df, features, clf))
        '''
        metric = []
        if cv_type=='cv_fix': #['feature_selection']['greedy']
            metric = self.cv_by_fix_folds(df, features, clf)
        elif cv_type=='train_test': #['feature_selection']['greedy']
            metric = self.train_test_clf(df, features, clf)
        elif cv_type=='cv_k_folds':
            metric = self.cv_k_folds_random(df, features, clf)
        elif cv_type=='cv_loo':
            metric = self.cv_loo(df, features, clf, path_to_save)
        return metric

    def cv_result(self, df, features, path_to_save, cv_type):
        clf_cv = []
        for clf in self.config['ml']['classifiers']:
            clf_cv.append(self.cv_result_clf(df, features, clf, cv_type, path_to_save))
        return pd.concat(clf_cv)

    def calculate_info_metric(self, test):
        result = {}
        for dataset in self.config['ml']['datasets']:
            res = test[test['dataset'] == dataset]
            result.update({dataset: round(accuracy_score(res['class'], res['pred']) * 100, 2),
                           'data_' + dataset: int(res.shape[0])})
        # print(pd.DataFrame(result))
        for stage in [0, 1, 2, 2.5, 3, 3.5]:
            res = test[test['stage'] == stage]
            result.update({'stage' + str(stage): round(accuracy_score(res['class'], res['pred']) * 100, 2),
                           'data_stage' + str(stage): int(res.shape[0])})
        # print(pd.DataFrame(result))
        for class_type in [0, 1, 2, 3]:
            res = test[test['class'] == class_type]
            result.update({'class' + str(class_type): round(accuracy_score(res['class'], res['pred']) * 100, 2),
                           'data_class' + str(class_type): int(res.shape[0])})
        # print(test[test['stage'] == 1][['pred', 'id']].groupby('id').mean())
        return result


    def processing(self, path_to_save):
        pd.set_option('display.width', 1000)
        pd.set_option('display.expand_frame_repr', False)

        #dataset correction with DI features and replace stages
        features = self.config['ml']['features']
        df = self.dataset_correction(features)

        #train_sets split result
        df_metric = self.train_test(df, features, path_to_save)
        df_metric.to_csv(os.path.join(path_to_save, 'metrics.csv'))
        print('TRAIN-TEST')
        print(df_metric[self.config['ml']['print_console']])

        #cross-validation result cv_fix
        df_cv_metric = self.cv_result(df, features, path_to_save, 'cv_fix')
        df_cv_metric.to_csv(os.path.join(path_to_save, 'cv_fix_metrics.csv'))
        print('CV FIX')
        print(df_cv_metric[[col + '±SD' for col in self.config['ml']['print_console']]])

        #cross-validation result cv_k_folds
        df_cv_metric = self.cv_result(df, features, path_to_save, 'cv_k_folds')
        df_cv_metric.to_csv(os.path.join(path_to_save, 'cv_k_folds_metrics.csv'))
        print('CV K-FOLDS')
        print(df_cv_metric[[col + '±SD' for col in self.config['ml']['print_console']]])

        #cross-validation result cv_loo
        df_cv_metric = self.cv_result(df, features, path_to_save, 'cv_loo')
        df_cv_metric.to_csv(os.path.join(path_to_save, 'cv_loo_metrics.csv'))
        print('CV LOO')
        print(df_cv_metric[[col + '±SD' for col in self.config['ml']['print_console']]])
        print(df_cv_metric[[col for col in df_cv_metric.columns if 'loo' in col]])


        #feature selection
        #self.greedy_forward_result(df, features)



