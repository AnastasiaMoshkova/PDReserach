import pandas as pd
import numpy as np
import os
import glob
from ml_cls.ml_base import MLBase
from hydra.utils import instantiate
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2, f_regression, f_classif, SelectPercentile
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

class GreedyAlgorithm(MLBase):

    def __init__(self, config):
        self.config = config

    def greedy_step(self, df, features, clf, prev_features, metric_optima, cv_type):
        metrics = []
        for feature in features:
            feature_vector_step = [feature]
            feature_vector_step.extend(prev_features)
            metric = self.cv_result_clf(df, feature_vector_step, clf, cv_type, '')
            metric['feature_add'] = feature
            metric['feature_vector'] = [feature_vector_step]
            metrics.append(metric)
        df_metrics = pd.concat(metrics)
        df_metrics = df_metrics.sort_values(metric_optima, ascending=False)
        #print(df_metrics)
        return df_metrics.iloc[0]
    def greedy_clf_forward(self, df, features, clf, metric_optima, cv_type):
        #copy all features elements
        features = features[:]
        res = []
        prev_features = []
        while len(features) != 0:
            step_result = self.greedy_step(df, features, clf, prev_features, metric_optima, cv_type)
            res.append(step_result)
            features.remove(step_result['feature_add'])
            prev_features = step_result['feature_vector']
        print(pd.concat(res, axis = 1).transpose())
        #print(pd.concat(res, axis = 1).transpose().sort_values(metric_optima, ascending=False))
        result = pd.concat(res, axis = 1).transpose().sort_values(metric_optima, ascending=False).iloc[0]
        path_to_save = r"C:\Users\kafed\Downloads\feature_save"
        pd.concat(res, axis = 1).transpose().to_csv(os.path.join(path_to_save, clf+'.csv'))
        return result

    def greedy_forward_result(self, df, features, cv_type):
        results = []
        metric_optima = self.config['ml']['metric'] #['feature_selection']['greedy']
        if ((metric_optima=='balanced_accuracy')&(self.config['ml']['track']=='cv_loo')):
            metric_optima = 'balanced_accuracy_loo'
        for clf in self.config['ml']['classifiers']:
            result = self.greedy_clf_forward(df, features, clf, metric_optima, cv_type)
            results.append(result)
            print(clf, result[metric_optima], result['feature_vector'])
        print(pd.concat(results, axis = 1).transpose())

    def remove_collinear_features(self, df, features, threshold):
        '''
        Objective:
            Remove collinear features in a dataframe with a correlation coefficient
            greater than the threshold. Removing collinear features can help a model
            to generalize and improves the interpretability of the model.

        Inputs:
            x: features dataframe
            threshold: features with correlations greater than this value are removed

        Output:
            dataframe that contains only the non-highly-collinear features
        '''

        # Calculate the correlation matrix
        x = df[features]
        corr_matrix = x.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        # Iterate through the correlation matrix and compare correlations
        for i in iters:
            for j in range(i + 1):
                item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
                col = item.columns
                row = item.index
                val = abs(item.values)

                # If correlation exceeds the threshold
                if val >= threshold:
                    # Print the correlated features and the correlation value
                    print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                    drop_cols.append(col.values[0])

        # Drop one of each pair of correlated columns
        drops = set(drop_cols)
        df = df.drop(columns=drops)

        features = list(set(features) - drops)

        print('##### feature after remove collinear features :')
        print('number :', len(features ))
        print('features :', features)

        return features


    def feature_filtering(self): #TODO
        pass

    def range_features(self): #TODO
        pass

    def remove_low_variance(self, df, features, threshold):
        df = self.data_processing_cv(df).dropna(subset=features)
        df[features] = df[features].replace({0: np.NaN, -1: np.NaN})
        sel = VarianceThreshold(threshold=threshold).set_output(transform="pandas")
        df_new = sel.fit_transform(df[features])
        print('##### feature after remove low variance :')
        print('number :', len(df_new.columns))
        print('features :', list(df_new.columns))
        return list(df_new.columns)


    def select_k_best(self, df_input, features, k, method):
        df = df_input.copy()
        df[features] = df[features].replace({0: np.NaN, -1: np.NaN})
        df = self.data_processing_cv(df).dropna(subset = features)
        if method == "chi2":
            selector = SelectKBest(chi2, k=k).set_output(transform="pandas")
        elif method == "f_regression":
            selector = SelectKBest(f_regression, k=k).set_output(transform="pandas")
        elif method == "f_classif":
            selector = SelectKBest(f_classif, k=k).set_output(transform="pandas")
        df_new = selector.fit_transform(df[features], df[self.config['ml']['target']])
        features = list(df_new.columns)

        print('##### feature after select k best by {} method:'.format(method))
        print('number :', len(df_new.columns))
        print('features :', list(df_new.columns))

        return features

    def giniBest_clf(self, df_input, features, number_persent = 0.15):
        forest = RandomForestClassifier(random_state=0)
        sample_weight = compute_sample_weight(class_weight="balanced", y=df_input[self.config['ml']['target']])
        forest.fit(df_input[features], df_input[self.config['ml']['target']], sample_weight = sample_weight)
        importances = forest.feature_importances_
        forest_importances = pd.Series(importances, index=features)
        number = round(forest_importances.shape[0] * number_persent)
        important_features = forest_importances.sort_values(ascending=False)[0:number]

        features = list(important_features.index)

        print('##### feature after select by Gini (RF) {} persent:'.format(number_persent * 100))
        print('number :', len(features))
        print('features :', features)

        return features


    def select_train(self, df_input, features):
        df_train, df_test = self.data_processing_train_test(df_input)
        print('##### used train data for feature selection: 0: {}, 1: {}, 2: {}, 3: {} '.format(
            df_train[df_train['stage']==0]['id'].unique().shape[0],
            df_train[df_train['stage'] == 1]['id'].unique().shape[0],
            df_train[df_train['stage'] == 2]['id'].unique().shape[0],
            df_train[df_train['stage'] == 3]['id'].unique().shape[0],
        ))
        return df_train

    def find_best_by_statistic(self, features):
        dfs = []
        for file in glob.glob(os.path.join(self.config['ml']['find_best_by_statistic'],'*.csv')):
            df_feature = pd.read_csv(os.path.join(self.config['ml']['find_best_by_statistic'], file))
            dfs.append(df_feature[['Unnamed: 0', 'anova_result']])
        df = pd.concat(dfs)
        df = df[df['Unnamed: 0'].isin(features)]
        features = list(df[df['anova_result']=='Different']['Unnamed: 0'])

        print('##### feature after select by statistic:')
        print('number :', len(features))
        print('features :', features)

        return features


    def find_best_by_statistic_progressive_early(self, features):
        new_features = []
        dfs = []
        for file in glob.glob(os.path.join(self.config['ml']['find_best_by_statistic'],'*.csv')):
            df_feature = pd.read_csv(os.path.join(self.config['ml']['find_best_by_statistic'], file))
            dfs.append(df_feature[['Unnamed: 0', 'mean0', 'mean12', 'mean3']])
        df = pd.concat(dfs)
        df = df[df['Unnamed: 0'].isin(features)]
        for feature in df['Unnamed: 0']:
            feature_means = df[df['Unnamed: 0']==feature]
            if ((feature_means['mean0'].values >= feature_means['mean12'].values) & (feature_means['mean12'].values >= feature_means['mean3'].values)):
                new_features.append(feature)
            if ((feature_means['mean3'].values >= feature_means['mean12'].values) & (feature_means['mean12'].values >= feature_means['mean0'].values)):
                new_features.append(feature)

        print('##### feature after select by statistic progressive early:')
        print('number :', len(new_features))
        print('features :', new_features)

        return new_features






    def processing(self, path_to_save):
        pd.set_option('display.width', 1000)
        pd.set_option('display.expand_frame_repr', False)

        #dataset correction with DI features and replace stages
        features = self.config['ml']['features']
        df = self.dataset_correction(features)

        #select only train users
        df = self.select_train(df, features)

        #features = self.feature_filtering(df, features)
        #features = self.range_features(df, features)

        #remove collinear features
        features = self.remove_collinear_features(df, features, self.config['ml']['correlation_threshold'])

        #remove low variance
        #features = self.remove_low_variance(df, features, 0.8) #TODO

        #select by statistic
        features = self.find_best_by_statistic(features)
        #features = self.find_best_by_statistic_progressive_early(features)

        #select k best features
        #features = self.select_k_best(df, features, self.config['ml']['k_best'], self.config['ml']['method_selection'])

        #range features by Gini
        #features = self.giniBest_clf(df, features, 0.5)

        # feature selection
        self.greedy_forward_result(df, features, self.config['ml']['track'])