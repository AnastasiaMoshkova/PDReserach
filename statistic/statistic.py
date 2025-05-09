import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import mannwhitneyu
import functools as ft
from scipy.stats import shapiro
import seaborn as sns
from scipy.stats import f_oneway
from latex import *
import json
import math
import glob
from functools import reduce
from hydra.utils import instantiate


class Statistic():
    def __init__(self, config):
        self.config = config

    def calculate_meta_statistic(self, data, output_dir):
        statsistic_meta_datsets = []
        for dataset in self.config['datasets']:
            df = data[data['dataset'] == dataset]
            statsistic_meta = {}
            statsistic_meta.update({'dataset': dataset})
            for parameter in ['gender', 'age', 'stage']:
                statsistic_meta.update({parameter:str(round(df.groupby('id')[parameter].mean().mean(),2)) + '±' + str(round(df.groupby('id')[parameter].mean().std(),2))})
            for stage in self.config['stages']:
                statsistic_meta.update({str(stage) + '_row': df[df['stage'] == stage].shape[0]})
                statsistic_meta.update({str(stage) + '_users': len(df[df['stage'] == stage]['id'].unique())})
            statsistic_meta.update({'total_row': df.shape[0]})
            statsistic_meta.update({'total_users': len(df['id'].unique())})
            statsistic_meta_datsets.append(statsistic_meta)
        pd.DataFrame(statsistic_meta_datsets).to_csv(os.path.join(output_dir, 'statistic_by_datsets.csv'))

        for dataset in ['PD', 'HEALTHY']:
            statsistic_meta = {}
            df_dataset = data[data['dataset'] == dataset]
            for stage in self.config['stages']:
                df = df_dataset[df_dataset['stage'] == stage]
                statsistic_meta.update({str(stage) : {'users': len(df['id'].unique()),
                                                      'age': str(round(df.groupby('id')['age'].mean().mean(),2)) + '±' + str(round(df.groupby('id')['age'].mean().std(),2)),
                                                      'male': df.groupby('id')['gender'].mean().sum(),
                                                      'male, %': round(df.groupby('id')['gender'].mean().sum() *100 /len(df['id'].unique()),2)}})
                statsistic_meta.update({str(stage) + '_row': {'users': df.shape[0],
                                                     'age': str(round(df['age'].mean(), 2)) + '±' + str(round(df['age'].std(), 2)),
                                                     'male': df['gender'].sum(),
                                                     'male, %': round(df['gender'].sum() * 100 / df.shape[0], 2)}})
            pd.DataFrame(statsistic_meta).to_csv(os.path.join(output_dir, dataset+'_statistic_by_stages.csv'))

    def _anova_test(self, dfs, features):
        anova_result = []
        for feature in features:
            statistic, pvalue = f_oneway(*dfs)
            anova_result.append(pd.DataFrame({'feature':feature, 'anova_stat': statistic, 'anova_p': pvalue}))
        return anova_result
    def plot_thetagrids_early(self, output_dir, mode, data):
        result = []
        df_stats = []
        anova_res = []
        kruskal_res = []
        for ex in self.config[mode]['exercise']:
            features = self.config[mode]['feature_type']
            if mode == 'em':
                features = [ex + '_' + feature for feature in features]
            elif mode == 'tremor':
                features = [feature + str(frq[0]) + str(frq[1]) + '_' + ex for feature in features for frq in self.config[mode]['frequency']]
            else:
                features = [feature + '_' + ex for feature in features]


            stage00 = data[data['stage'] == 0][features].replace(-1, np.NaN)#.replace(0, np.NaN)
            stage12 = data[data['stage'].isin([1,2])][features].replace(-1, np.NaN)#.replace(0, np.NaN)
            stage3 = data[data['stage'].isin([3])][features].replace(-1, np.NaN)#.replace(0, np.NaN)


            anova_result = []
            kruskal_result = []

            features_name = []
            for feature in features:
                statistic, pvalue = f_oneway(stage00.dropna()[feature],stage12.dropna()[feature],stage3.dropna()[feature])
                if pvalue<0.05:
                    res = 'Different'
                else:
                    res = 'Same'
                anova_result.append({'feature': feature, 'anova_stat': statistic, 'anova_p': pvalue, 'anova_result': res})

                statistic, pvalue = stats.kruskal(stage00.dropna()[feature],stage12.dropna()[feature],stage3.dropna()[feature])
                if pvalue<0.05:
                    res = 'Different'
                else:
                    res = 'Same'
                kruskal_result.append({'feature': feature, 'kruskal_stat': statistic, 'kruskal_p': pvalue, 'kruskal_result': res})

                stat, p1 = shapiro(stage00.dropna()[feature])
                stat, p2 = shapiro(stage12.dropna()[feature])
                stat, p3 = shapiro(stage3.dropna()[feature])
                if ((p1>0.05) & (p2>0.05) & (p3>0.05)):
                    statistic, pvalue = f_oneway(stage00.dropna()[feature], stage12.dropna()[feature],stage3.dropna()[feature])
                    if pvalue < 0.05:
                        features_name.append(feature + '$^*$')
                    else:
                        features_name.append(feature)
                else:
                    statistic, pvalue = stats.kruskal(stage00.dropna()[feature], stage12.dropna()[feature], stage3.dropna()[feature])
                    if pvalue < 0.05:
                        features_name.append(feature + '$^*$')
                    else:
                        features_name.append(feature)


            anova = pd.DataFrame(anova_result)
            anova.index = anova['feature']
            anova_res.append(anova)

            kruskal = pd.DataFrame(kruskal_result)
            kruskal.index = kruskal['feature']
            kruskal_res.append(kruskal)

            df00 = self.calculate_statistic(stage00, '0')
            df12 = self.calculate_statistic(stage12, '12')
            df3 = self.calculate_statistic(stage3, '3')
            result.append(pd.concat([df00, df12, df3], axis = 1))

            df_stat = []
            for critery in self.config['stat_critery']:
                df_stat.append(pd.DataFrame(self.calculate_statistic_differense(stage00, stage12, critery, '0-12')))
                df_stat.append(pd.DataFrame(self.calculate_statistic_differense(stage12, stage3, critery, '12-3')))
                df_stat.append(pd.DataFrame(self.calculate_statistic_differense(stage00, stage3, critery, '0-3')))

            df_stat = ft.reduce(lambda left, right: pd.merge(left, right, on='feature'), df_stat)
            df_stat.index = df_stat['feature']
            df_stats.append(df_stat)

            stage00 = stage00.describe().loc[self.config['aggregation_type']].values
            stage12 = stage12.describe().loc[self.config['aggregation_type']].values
            stage3 = stage3.describe().loc[self.config['aggregation_type']].values

            stage0 = list(stage00 / stage00)
            stage12 = list(stage12 / stage00)
            stage3 = list(stage3 / stage00)

            stage0.append(stage0[0])
            stage12.append(stage12[0])
            stage3.append(stage3[0])

            if ex == 'FT':
                plt.rcParams.update({'font.size': 16})
            else:
                plt.rcParams.update({'font.size': 16})

            #plt.figure(figsize=(12, 10))
            plt.figure(figsize=(15, 10))
            plt.subplot(polar=True)

            theta = np.linspace(0, 2 * np.pi, len(stage0))
            #plt.rcParams["text.usetex"] = True
            #features = [feature + '$^*$' for feature in features]
            #lines, labels = plt.thetagrids(range(0, 360, int(360 / len(features))), (features))

            # Plot actual sales graph
            plt.plot(theta, stage0, linewidth = 5.0)
            plt.plot(theta, stage12, linewidth = 5.0)
            plt.plot(theta, stage3, linewidth=5.0)

            lines, labels = plt.thetagrids(range(0, 360, int(360 / len(features))), (features_name))

            #plt.legend(labels=('stage0', 'stage12', 'stage3'), loc='best',bbox_to_anchor=(0.65, 0.3, 0.6, 0.5))
            plt.legend(labels=('Healthy', 'Early', 'Middle'), loc='best', bbox_to_anchor=(0.8, 0.3, 0.6, 0.5))
            #plt.title(ex)
            plt.savefig(os.path.join(output_dir, ex + '.png'))

        df = pd.concat(result)
        df_stats = pd.concat(df_stats)
        df_anova = pd.concat(anova_res)
        df_kruskal = pd.concat(kruskal_res)
        df = pd.merge(df, df_stats, left_index=True, right_index=True)
        df = pd.merge(df, df_anova, left_index=True, right_index=True)
        df = pd.merge(df, df_kruskal, left_index=True, right_index=True)
        #columns = ['M ± SD_0','M ± SD_12', 't-test p-value 0-12', 't-test 0-12', 'Mann-W p-value 0-12', 'Mann-W 0-12','mean0', 'std0', 'median0', 'mean12', 'std12', 'median12']
        df.to_csv(os.path.join(output_dir, mode + '_early_statistic.csv'))
    def plot_thetagrids_binary(self, output_dir, mode, data):
        result = []
        df_stats = []
        anova_res = []
        for ex in self.config[mode]['exercise']:
            features = self.config[mode]['feature_type']
            if mode == 'em':
                features = [ex + '_' + feature for feature in features]
            elif mode == 'tremor':
                features = [feature + str(frq[0]) + str(frq[1]) + '_' + ex for feature in features for frq in self.config[mode]['frequency']]
            else:
                features = [feature + '_' + ex for feature in features]

            stage00 = data[data['stage'] == 0][features].replace(-1, np.NaN)#.replace(0, np.NaN)
            stage123 = data[data['stage'].isin([1,2,3])][features].replace(-1, np.NaN)#.replace(0, np.NaN)

            anova_result = []
            for feature in features:
                statistic, pvalue = f_oneway(stage00.dropna()[feature], stage123.dropna()[feature])
                if pvalue < 0.05:
                    res = 'Different'
                else:
                    res = 'Same'
                anova_result.append(
                    {'feature': feature, 'anova_stat': statistic, 'anova_p': pvalue, 'anova_result': res})

            anova = pd.DataFrame(anova_result)
            anova.index = anova['feature']
            anova_res.append(anova)

            df00 = self.calculate_statistic(stage00, '0')
            df123 = self.calculate_statistic(stage123, '123')
            result.append(pd.concat([df00, df123], axis = 1))

            df_stat = []
            for critery in self.config['stat_critery']:
                df_diff = self.calculate_statistic_differense(stage00, stage123, critery, '0-123')
                df_stat.append(pd.DataFrame(df_diff))
            df_stat = ft.reduce(lambda left, right: pd.merge(left, right, on='feature'), df_stat)
            df_stat.index = df_stat['feature']
            df_stats.append(df_stat)

            stage00 = stage00.describe().loc[self.config['aggregation_type']].values
            stage123 = stage123.describe().loc[self.config['aggregation_type']].values

            stage0 = list(stage00 / stage00)
            stage123 = list(stage123 / stage00)

            stage0.append(stage0[0])
            stage123.append(stage123[0])

            plt.figure(figsize=(12, 10))
            plt.subplot(polar=True)

            theta = np.linspace(0, 2 * np.pi, len(stage0))

            lines, labels = plt.thetagrids(range(0, 360, int(360 / len(features))), (features))

            # Plot actual sales graph
            plt.plot(theta, stage0, linewidth = 2.0)
            plt.plot(theta, stage123, linewidth = 2.0)

            plt.legend(labels=('stage0', 'stage123'), loc='best',bbox_to_anchor=(0.65, 0.3, 0.6, 0.5))
            plt.title(ex)
            plt.savefig(os.path.join(output_dir, ex + '.png'))
        df = pd.concat(result)
        df_stats = pd.concat(df_stats)
        df_anova = pd.concat(anova_res)
        df = pd.merge(df, df_stats, left_index=True, right_index=True)
        df = pd.merge(df, df_anova, left_index=True, right_index=True)
        #columns = ['M ± SD_0','M ± SD_123', 't-test p-value 0-123', 't-test 0-123', 'Mann-W p-value 0-123', 'Mann-W 0-123','mean0', 'std0', 'median0', 'mean123', 'std123', 'median123']
        #df[columns].to_csv(os.path.join(output_dir, mode + '_binary_statistic.csv'))
        df.to_csv(os.path.join(output_dir, mode + '_binary_statistic.csv'))
    def plot_thetagrids_multistage(self, output_dir, mode, data):
        result = []
        df_stats = []
        anova_res = []
        for ex in self.config[mode]['exercise']:
            features = self.config[mode]['feature_type']
            if mode == 'em':
                features = [ex + '_' + feature for feature in features]
            elif mode == 'tremor':
                features = [feature + str(frq[0]) + str(frq[1]) + '_' + ex for feature in features for frq in self.config[mode]['frequency']]
            else:
                features = [feature + '_' + ex for feature in features]

            stage00 = data[data['stage'] == 0][features].replace(-1, np.NaN)#.replace(0, np.NaN) #.replace([np.inf, -np.inf], np.NaN)
            stage1 = data[data['stage'] == 1][features].replace(-1, np.NaN)#.replace(0, np.NaN) #.replace([np.inf, -np.inf], np.NaN)
            stage2 = data[data['stage'] == 2][features].replace(-1, np.NaN)#.replace(0, np.NaN) #.replace([np.inf, -np.inf], np.NaN)
            stage3 = data[data['stage'] == 3][features].replace(-1, np.NaN)#.replace(0, np.NaN) #.replace([np.inf, -np.inf], np.NaN)

            anova_result = []
            for feature in features:
                statistic, pvalue = f_oneway(stage00.dropna()[feature], stage1.dropna()[feature], stage2.dropna()[feature], stage3.dropna()[feature])
                if pvalue < 0.05:
                    res = 'Different'
                else:
                    res = 'Same'
                anova_result.append(
                    {'feature': feature, 'anova_stat': statistic, 'anova_p': pvalue, 'anova_result': res})
            anova = pd.DataFrame(anova_result)
            anova.index = anova['feature']
            anova_res.append(anova)

            df00 = self.calculate_statistic(stage00, '0')
            df1 = self.calculate_statistic(stage1, '1')
            df2 = self.calculate_statistic(stage2, '2')
            df3 = self.calculate_statistic(stage3, '3')
            result.append(pd.concat([df00, df1, df2, df3], axis=1))

            df_stat = []
            for critery in self.config['stat_critery']:
                df_stat.append(pd.DataFrame(self.calculate_statistic_differense(stage00, stage1, critery, '0-1')))
                df_stat.append(pd.DataFrame(self.calculate_statistic_differense(stage00, stage2, critery, '0-2')))
                df_stat.append(pd.DataFrame(self.calculate_statistic_differense(stage00, stage3, critery, '0-3')))
                df_stat.append(pd.DataFrame(self.calculate_statistic_differense(stage1, stage2, critery, '1-2')))
                df_stat.append(pd.DataFrame(self.calculate_statistic_differense(stage1, stage3, critery, '1-3')))
                df_stat.append(pd.DataFrame(self.calculate_statistic_differense(stage2, stage3, critery, '2-3')))
            df_stat = ft.reduce(lambda left, right: pd.merge(left, right, on='feature'), df_stat)
            df_stat.index = df_stat['feature']
            df_stats.append(df_stat)

            stage00 = stage00.describe().loc[self.config['aggregation_type']].values
            stage1 = stage1.describe().loc[self.config['aggregation_type']].values
            stage2 = stage2.describe().loc[self.config['aggregation_type']].values
            stage3 = stage3.describe().loc[self.config['aggregation_type']].values

            stage0 = list(stage00 / stage00)
            stage1 = list(stage1 / stage00)
            stage2 = list(stage2 / stage00)
            stage3 = list(stage3 / stage00)

            stage0.append(stage0[0])
            stage1.append(stage1[0])
            stage2.append(stage2[0])
            stage3.append(stage3[0])

            plt.figure(figsize=(12, 10))
            plt.subplot(polar=True)

            theta = np.linspace(0, 2 * np.pi, len(stage0))

            lines, labels = plt.thetagrids(range(0, 360, int(360 / len(features))), (features))

            # Plot actual sales graph
            plt.plot(theta, stage0, linewidth = 2.0)
            plt.plot(theta, stage1, linewidth = 2.0)
            plt.plot(theta, stage2, linewidth = 2.0)
            plt.plot(theta, stage3, linewidth = 2.0)

            plt.legend(labels=('stage0', 'stage1', 'stage2', 'stage3'), loc='best',bbox_to_anchor=(0.65, 0.3, 0.6, 0.5))
            plt.title(ex)
            plt.savefig(os.path.join(output_dir, ex+'.png'))
        df = pd.concat(result)
        df_stats = pd.concat(df_stats)
        df_anova = pd.concat(anova_res)
        df = pd.merge(df, df_stats, left_index=True, right_index=True)
        df = pd.merge(df, df_anova, left_index=True, right_index=True)
        #columns = ['M ± SD_0','M ± SD_123', 't-test p-value 0-123', 't-test 0-123', 'Mann-W p-value 0-123', 'Mann-W 0-123','mean0', 'std0', 'median0', 'mean123', 'std123', 'median123']
        df.to_csv(os.path.join(output_dir, mode + '_multistages_statistic.csv'))

    def hand_tremor_dataset_processing(self, df):
        dfL = df[df['hand'] == 'L']
        dfR = df[df['hand'] == 'R']
        dfR = dfR.drop(columns=['3.4b_FT','3.5b_OC','3.6b_PS', '3.15b_Постуральный тремор','3.16b_Кинетический тремор','3.17b_Амплитуда термора покоя', ])
        dfL = dfL.drop(columns=['3.4a_FT','3.5a_OC','3.6a_PS', '3.15a_Постуральный тремор','3.16a_Кинетический тремор','3.17a_Амплитуда термора покоя',])
        dfR = dfR.rename(columns={'3.4a_FT': '3.4_FT',
                                  '3.5a_OC': '3.5_OC',
                                  '3.6a_PS': '3.6_PS',
                                  '3.15a_Постуральный тремор': '3.15_Постуральный тремор',
                                  '3.16a_Кинетический тремор': '3.16_Кинетический тремор',
                                  '3.17a_Амплитуда термора покоя': '3.17_Амплитуда термора покоя',
                                  })
        dfL = dfL.rename(columns={'3.4b_FT': '3.4_FT',
                                  '3.5b_OC': '3.5_OC',
                                  '3.6b_PS': '3.6_PS',
                                  '3.15b_Постуральный тремор': '3.15_Постуральный тремор',
                                  '3.16b_Кинетический тремор': '3.16_Кинетический тремор',
                                  '3.17b_Амплитуда термора покоя': '3.17_Амплитуда термора покоя',
                                  })
        dfRL = pd.concat([dfR, dfL])
        dfRL['3.6_PS'] = dfRL['3.6_PS'].replace({0.5: 0})

        return dfRL

    def plot_thetagrids_MDS_UPDRS(self, output_dir, mode, data_init):
        data = data_init.copy()

        if ((mode == 'hand') | (mode == 'tremor')):
            data = self.hand_tremor_dataset_processing(data)

        for ex in self.config[mode]['exercise']:
            features = self.config[mode]['feature_type']
            if mode == 'em':
                features = [ex + '_' + feature for feature in features]
            elif mode == 'tremor':
                features = [feature + str(frq[0]) + str(frq[1]) + '_' + ex for feature in features for frq in self.config[mode]['frequency']]
            else:
                features = [feature + '_' + ex for feature in features]

            if (mode == 'hand'):
                mds_updrs_cols = self.config[mode]['MDS-UPDRS'][ex]
            else:
                mds_updrs_cols = self.config[mode]['MDS-UPDRS']

            for mds_updrs_col in mds_updrs_cols:

                result = []
                df_stats = []
                anova_res = []
                kruskal_res = []

                stage00 = data[(data[mds_updrs_col] == 0) & (data['dataset']!='PD')][features].replace(-1, np.NaN)#.replace(0, np.NaN)
                stage0 = data[(data[mds_updrs_col] == 0) & (data['dataset']=='PD')][features].replace(-1, np.NaN)#.replace(0, np.NaN)
                stage1 = data[(data[mds_updrs_col] == 1) & (data['dataset']=='PD')][features].replace(-1, np.NaN)  # .replace(0, np.NaN)
                stage2 = data[(data[mds_updrs_col] == 2) & (data['dataset']=='PD')][features].replace(-1, np.NaN)#.replace(0, np.NaN)
                stage3 = data[(data[mds_updrs_col] == 3) & (data['dataset'] == 'PD')][features].replace(-1,np.NaN)  # .replace(0, np.NaN)
                stage4 = data[(data[mds_updrs_col] == 4) & (data['dataset'] == 'PD')][features].replace(-1, np.NaN)  # .replace(0, np.NaN)


                anova_result = []
                kruskal_result = []

                features_name = []
                for feature in features:
                    statistic, pvalue = f_oneway(stage0.dropna()[feature],
                                                 stage1.dropna()[feature],
                                                 stage2.dropna()[feature],
                                                 stage3.dropna()[feature],
                                                 #stage4.dropna()[feature]
                                                 )
                    if pvalue<0.05:
                        res = 'Different'
                    else:
                        res = 'Same'
                    anova_result.append({'feature': feature, 'anova_stat': statistic, 'anova_p': pvalue, 'anova_result': res})

                    statistic, pvalue = stats.kruskal(stage0.dropna()[feature],
                                                      stage1.dropna()[feature],
                                                      stage2.dropna()[feature],
                                                      stage3.dropna()[feature],
                                                      #stage4.dropna()[feature]
                                                      )
                    if pvalue<0.05:
                        res = 'Different'
                    else:
                        res = 'Same'
                    kruskal_result.append({'feature': feature, 'kruskal_stat': statistic, 'kruskal_p': pvalue, 'kruskal_result': res})

                    stat, p0 = shapiro(stage0.dropna()[feature])
                    stat, p1 = shapiro(stage1.dropna()[feature])
                    stat, p2 = shapiro(stage2.dropna()[feature])
                    stat, p3 = shapiro(stage3.dropna()[feature])
                    #stat, p4 = shapiro(stage4.dropna()[feature])
                    if ((p0>0.05) & (p1>0.05) & (p2>0.05) & (p3>0.05)): #& (p4>0.05)):
                        statistic, pvalue = f_oneway(stage0.dropna()[feature],
                                                      stage1.dropna()[feature],
                                                      stage2.dropna()[feature],
                                                      stage3.dropna()[feature],
                                                      #stage4.dropna()[feature]
                                                     )
                        if pvalue < 0.05:
                            features_name.append(feature + '$^*$')
                        else:
                            features_name.append(feature)
                    else:
                        statistic, pvalue = stats.kruskal(stage0.dropna()[feature],
                                                      stage1.dropna()[feature],
                                                      stage2.dropna()[feature],
                                                      stage3.dropna()[feature],
                                                      #stage4.dropna()[feature]
                                                          )
                        if pvalue < 0.05:
                            features_name.append(feature + '$^*$')
                        else:
                            features_name.append(feature)


                anova = pd.DataFrame(anova_result)
                anova.index = anova['feature']
                anova_res.append(anova)

                kruskal = pd.DataFrame(kruskal_result)
                kruskal.index = kruskal['feature']
                kruskal_res.append(kruskal)

                df00 = self.calculate_statistic(stage00, '00')
                df0 = self.calculate_statistic(stage0, '0')
                df1 = self.calculate_statistic(stage1, '1')
                df2 = self.calculate_statistic(stage2, '2')
                df3 = self.calculate_statistic(stage3, '3')
                #df4 = self.calculate_statistic(stage4, '4')
                #result.append(pd.concat([df00, df0, df1, df2, df3, df4], axis = 1))
                result.append(pd.concat([df00, df0, df1, df2, df3], axis=1))

                df_stat = []
                for critery in self.config['stat_critery']:
                    df_stat.append(pd.DataFrame(self.calculate_statistic_differense(stage0, stage1, critery, '0-1')))
                    df_stat.append(pd.DataFrame(self.calculate_statistic_differense(stage1, stage2, critery, '1-2')))
                    df_stat.append(pd.DataFrame(self.calculate_statistic_differense(stage2, stage3, critery, '2-3')))

                df_stat = ft.reduce(lambda left, right: pd.merge(left, right, on='feature'), df_stat)
                df_stat.index = df_stat['feature']
                df_stats.append(df_stat)

                stage0h = stage00.describe().loc[self.config['aggregation_type']].values
                stage0 = stage0.describe().loc[self.config['aggregation_type']].values
                stage1 = stage1.describe().loc[self.config['aggregation_type']].values
                stage2 = stage2.describe().loc[self.config['aggregation_type']].values
                stage3 = stage3.describe().loc[self.config['aggregation_type']].values
                stage4 = stage4.describe().loc[self.config['aggregation_type']].values

                stage00 = list(stage0h / stage0h)
                stage0 = list(stage0 / stage0h)
                stage1 = list(stage1 / stage0h)
                stage2 = list(stage2 / stage0h)
                stage3 = list(stage3 / stage0h)
                stage4 = list(stage4 / stage0h)

                stage00.append(stage00[0])
                stage0.append(stage0[0])
                stage1.append(stage1[0])
                stage2.append(stage2[0])
                stage3.append(stage3[0])
                stage4.append(stage4[0])

                if ex == 'FT':
                    plt.rcParams.update({'font.size': 16})
                else:
                    plt.rcParams.update({'font.size': 16})

                #plt.figure(figsize=(12, 10))
                plt.figure(figsize=(15, 10))
                plt.subplot(polar=True)

                theta = np.linspace(0, 2 * np.pi, len(stage0))
                #plt.rcParams["text.usetex"] = True
                #features = [feature + '$^*$' for feature in features]
                #lines, labels = plt.thetagrids(range(0, 360, int(360 / len(features))), (features))

                # Plot actual sales graph
                plt.plot(theta, stage00, linewidth = 5.0)
                plt.plot(theta, stage0, linewidth=5.0)
                plt.plot(theta, stage1, linewidth = 5.0)
                plt.plot(theta, stage2, linewidth=5.0)
                plt.plot(theta, stage3, linewidth=5.0)
                #plt.plot(theta, stage4, linewidth=5.0)

                lines, labels = plt.thetagrids(range(0, 360, int(360 / len(features))), (features_name))

                #plt.legend(labels=('stage0', 'stage12', 'stage3'), loc='best',bbox_to_anchor=(0.65, 0.3, 0.6, 0.5))
                #plt.legend(labels=('Healthy', 'UPDRS0', 'UPDRS1','UPDRS2','UPDRS3','UPDRS4'), loc='best', bbox_to_anchor=(0.8, 0.3, 0.6, 0.5))
                plt.legend(labels=('Healthy', 'UPDRS0', 'UPDRS1', 'UPDRS2', 'UPDRS3'), loc='best',
                           bbox_to_anchor=(0.8, 0.3, 0.6, 0.5))
                #plt.title(ex)
                plt.savefig(os.path.join(output_dir, ex + '_' + mds_updrs_col+'_updrs.png'))

            df = pd.concat(result)
            df_stats = pd.concat(df_stats)
            df_anova = pd.concat(anova_res)
            df_kruskal = pd.concat(kruskal_res)
            df = pd.merge(df, df_stats, left_index=True, right_index=True)
            df = pd.merge(df, df_anova, left_index=True, right_index=True)
            df = pd.merge(df, df_kruskal, left_index=True, right_index=True)
            #columns = ['M ± SD_0','M ± SD_12', 't-test p-value 0-12', 't-test 0-12', 'Mann-W p-value 0-12', 'Mann-W 0-12','mean0', 'std0', 'median0', 'mean12', 'std12', 'median12']
            df.to_csv(os.path.join(output_dir, mode + '_' + mds_updrs_col + '_updrs_statistic.csv'))

    def plot_thetagrids_MDS_UPDRS2(self, output_dir, mode, data_init):
        data = data_init.copy()

        if ((mode == 'hand') | (mode == 'tremor')):
            data = self.hand_tremor_dataset_processing(data)

        for ex in self.config[mode]['exercise']:
            features = self.config[mode]['feature_type']
            if mode == 'em':
                features = [ex + '_' + feature for feature in features]
            elif mode == 'tremor':
                features = [feature + str(frq[0]) + str(frq[1]) + '_' + ex for feature in features for frq in self.config[mode]['frequency']]
            else:
                features = [feature + '_' + ex for feature in features]

            if (mode == 'hand'):
                mds_updrs_cols = self.config[mode]['MDS-UPDRS'][ex]
            else:
                mds_updrs_cols = self.config[mode]['MDS-UPDRS']

            for mds_updrs_col in mds_updrs_cols:

                stage00 = data[(data[mds_updrs_col] == 0) & (data['dataset']!='PD')][features].replace(-1, np.NaN)#.replace(0, np.NaN)
                stage0 = data[(data[mds_updrs_col] == 0) & (data['dataset']=='PD')][features].replace(-1, np.NaN)#.replace(0, np.NaN)
                stage1 = data[(data[mds_updrs_col] == 1) & (data['dataset']=='PD')][features].replace(-1, np.NaN)  # .replace(0, np.NaN)
                stage2 = data[(data[mds_updrs_col] == 2) & (data['dataset']=='PD')][features].replace(-1, np.NaN)#.replace(0, np.NaN)
                stage3 = data[(data[mds_updrs_col] == 3) & (data['dataset'] == 'PD')][features].replace(-1,np.NaN)  # .replace(0, np.NaN)
                stage4 = data[(data[mds_updrs_col] == 4) & (data['dataset'] == 'PD')][features].replace(-1, np.NaN)  # .replace(0, np.NaN)

                stage0h = stage00.describe().loc[self.config['aggregation_type']].values
                stage0 = stage0.describe().loc[self.config['aggregation_type']].values
                stage1 = stage1.describe().loc[self.config['aggregation_type']].values
                stage2 = stage2.describe().loc[self.config['aggregation_type']].values
                stage3 = stage3.describe().loc[self.config['aggregation_type']].values
                stage4 = stage4.describe().loc[self.config['aggregation_type']].values

                stage00 = list(stage0h / stage0h)
                stage0 = list(stage0 / stage0h)
                stage1 = list(stage1 / stage0h)
                stage2 = list(stage2 / stage0h)
                stage3 = list(stage3 / stage0h)
                stage4 = list(stage4 / stage0h)

                stage00.append(stage00[0])
                stage0.append(stage0[0])
                stage1.append(stage1[0])
                stage2.append(stage2[0])
                stage3.append(stage3[0])
                stage4.append(stage4[0])

                if ex == 'FT':
                    plt.rcParams.update({'font.size': 16})
                else:
                    plt.rcParams.update({'font.size': 16})

                #plt.figure(figsize=(12, 10))
                plt.figure(figsize=(15, 10))
                plt.subplot(polar=True)

                theta = np.linspace(0, 2 * np.pi, len(stage0))

                # Plot actual sales graph
                plt.plot(theta, stage00, linewidth = 5.0)
                plt.plot(theta, stage0, linewidth=5.0)
                plt.plot(theta, stage1, linewidth = 5.0)
                plt.plot(theta, stage2, linewidth=5.0)
                plt.plot(theta, stage3, linewidth=5.0)
                #plt.plot(theta, stage4, linewidth=5.0)

                lines, labels = plt.thetagrids(range(0, 360, int(360 / len(features))), (features))

                #plt.legend(labels=('stage0', 'stage12', 'stage3'), loc='best',bbox_to_anchor=(0.65, 0.3, 0.6, 0.5))
                #plt.legend(labels=('Healthy', 'UPDRS0', 'UPDRS1','UPDRS2','UPDRS3','UPDRS4'), loc='best', bbox_to_anchor=(0.8, 0.3, 0.6, 0.5))
                plt.legend(labels=('Healthy', 'UPDRS0', 'UPDRS1', 'UPDRS2', 'UPDRS3'), loc='best',
                           bbox_to_anchor=(0.8, 0.3, 0.6, 0.5))
                #plt.title(ex)
                plt.savefig(os.path.join(output_dir, ex + '_' + mds_updrs_col+'_updrs.png'))

    def save_plots(self, data, output_dir, mode):
        output_dir_multistage = os.path.join(output_dir, 'multistage_plot')
        if not os.path.isdir(output_dir_multistage):
            os.mkdir(output_dir_multistage)
        self.plot_thetagrids_multistage(output_dir_multistage, mode, data)

        output_dir_binary = os.path.join(output_dir, 'binary_plot')
        if not os.path.isdir(output_dir_binary):
            os.mkdir(output_dir_binary)
        self.plot_thetagrids_binary(output_dir_binary, mode, data)

        output_dir_early = os.path.join(output_dir, 'early_plot')
        if not os.path.isdir(output_dir_early):
            os.mkdir(output_dir_early)
        self.plot_thetagrids_early(output_dir_early, mode, data)

        output_dir_updrs = os.path.join(output_dir, 'mds-updrs')
        if not os.path.isdir(output_dir_updrs):
            os.mkdir(output_dir_updrs)
        self.plot_thetagrids_MDS_UPDRS2(output_dir_updrs, mode, data)


    def calculate_statistic(self, df_stage, stage):
        df = pd.DataFrame(df_stage.describe().loc[['mean', 'std', '50%']]).round(2).transpose()
        df = df.rename(columns={'mean': 'mean'+stage, 'std': 'std'+stage, '50%': 'median'+stage})
        df['M ± SD_'+stage] = df['mean'+stage].apply(str) + ' ± ' + df['std'+stage].apply(str)
        normal_res = self.normal_test(df_stage, stage)
        df = df.merge(normal_res, left_index = True, right_index = True)
        return df

    def mann(self, data1, data2, alpha):
        stat, p = mannwhitneyu(data1, data2)
        if p > alpha:
            st = 'Same'
        else:
            st = 'Different'
        return stat, p, st

    def ttest(self, data1, data2, alpha):
        stat, p = stats.ttest_ind(data1, data2, equal_var=True)
        if p > alpha:
            st = 'Same'
        else:
            st = 'Different'
        return stat, p, st

    def calculate_statistic_differense(self, df1, df2, critery, stage):
        result = []
        if critery == 'mann':
            for column in df1.columns:
                stat, p, st = self.mann(df1[column].dropna().values, df2[column].dropna().values, 0.05)
                result.append({'Mann-W p-value '+stage: p, 'Mann-W '+stage: st, 'feature': column, 'number_df1 '+stage: len(df1[column].dropna().values), 'number_df2 '+stage: len(df2[column].dropna().values)})
        if critery == 't-test':
            for column in df1.columns:
                stat, p, st = self.ttest(df1[column].dropna().values, df2[column].dropna().values, 0.05)
                result.append({'t-test p-value '+stage: p, 't-test '+stage: st, 'feature': column, 'number_df1 '+stage: len(df1[column].dropna().values), 'number_df2 '+stage: len(df2[column].dropna().values)})
        return result

    def normal_test(self, df, stage): #TODO
        result = []
        for column in df.columns:
            stat, p = shapiro(df[column].dropna().values)
            alpha = 0.05
            if p > alpha:
                res = True
            else:
                res = False
            result.append(pd.DataFrame({'Shapiro p-value ' + stage: p, 'Shapiro stat ' + stage: stat, 'number_Spapiro_stage' + stage: len(df[column].dropna().values), 'Gaussian': res}, index = [column]))
        return pd.concat(result)


    def correlation_UPDRS(self, data, features, features_correlate, columns_for_correlate, datasets, ex, mode):
        df = data[data['dataset'].isin(datasets)]
        if mode=='hand':
            df = df.groupby(['id', 'r', 'm'])[features_correlate].mean()
        df = df.dropna(subset=['stage', 'UPDRS_HAND_FACE', 'UPDRS_mimic'])
        df_spearman = pd.DataFrame(df[features_correlate].corr(method='spearman')[columns_for_correlate].dropna().loc[features])
        df_pearson = pd.DataFrame(df[features_correlate].corr()[columns_for_correlate].dropna().loc[features])
        df_spearman.columns = [col + '_spearman' for col in columns_for_correlate]
        df_pearson.columns = [col + '_pearson' for col in columns_for_correlate]
        df = df_spearman.merge(df_pearson, left_index=True, right_index=True)
        df['ex'] = ex
        return df

    def correlation_UPDRS_Hand(self, data, features, datasets, ex, output_dir):
        scores = {'R':['3.4a_FT','3.5a_OC','3.6a_PS'], 'L':['3.4b_FT','3.5b_OC','3.6b_PS']}
        df_spearman = []
        for hand in ['R', 'L']:
            features_correlate = []
            features_correlate.extend(scores[hand])
            features_correlate.extend(features)
            df = data[(data['dataset'].isin(datasets)) & (data['hand']==hand)]
            df = df.dropna(subset=scores[hand])
            df_spearman.append(pd.DataFrame(df[features_correlate].corr(method='spearman')[scores[hand]].dropna().loc[features]))
        df = pd.concat(df_spearman, axis=1)
        df['ex'] = ex
        df.to_csv(os.path.join(output_dir, 'correlation', ex+"_hand_RL_correlation.csv"))


    def correlation_matrix(self, data, output_dir, mode):
        df_pearson = []
        df_spearman = []
        df_hand = []
        df_hand_face = []
        columns_for_correlate = ['stage', 'UPDRS_3', 'UPDRS_1_2_4', 'UPDRS', 'UPDRS_HAND_FACE', 'UPDRS_mimic', '3.4a_FT','3.4b_FT','3.5a_OC', '3.5b_OC', '3.6a_PS', '3.6b_PS']
        if not os.path.isdir(os.path.join(output_dir,'correlation')):
            os.mkdir(os.path.join(output_dir,'correlation'))
        for ex in self.config[mode]['exercise']:
            features = self.config[mode]['feature_type']

            if mode == 'em':
                features = [ex + '_' + feature for feature in features]
            elif mode == 'tremor':
                features = [feature + str(frq[0]) + str(frq[1]) + '_' + ex for feature in features for frq in
                            self.config[mode]['frequency']]
            else:
                features = [feature + '_' + ex for feature in features]

            data[features] = data[features].replace(0, np.NaN).replace(-1, np.NaN)
            corr = data[features].corr()
            mask = np.triu(np.ones_like(corr, dtype = bool))
            cmap = sns.diverging_palette(100, 7, s=75, l=40, n=5, center="light", as_cmap = True)
            plt.figure(figsize=(12,12))
            sns_fig = sns.heatmap(corr, mask=mask, center=0, annot=True, fmt='.2f', square=True, cmap=cmap)
            fig = sns_fig.get_figure()
            fig.savefig(os.path.join(output_dir,'correlation', mode + '_' + ex + '_corr.png'))

            features_correlate = []
            features_correlate.extend(columns_for_correlate)
            features_correlate.extend(features)

            if mode == 'hand':
                self.correlation_UPDRS_Hand(data, features, ['PD'], ex, output_dir)
            df_hand_face.append(self.correlation_UPDRS(data, features, features_correlate, columns_for_correlate, ['PD'], ex, mode))

            data_corr = data[data['dataset'].isin(self.config['datasets'])][features_correlate]
            prsn_corr = pd.DataFrame(data_corr.corr()[columns_for_correlate].loc[features])
            prsn_corr.columns = [col + '_pearson' for col in columns_for_correlate]
            df_pearson.append(prsn_corr)
            sprm_corr = pd.DataFrame(data_corr.corr(method='spearman')[columns_for_correlate].loc[features])
            sprm_corr.columns = [col + '_spearman' for col in columns_for_correlate]
            df_spearman.append(sprm_corr)

            df_filtr = sprm_corr[abs(sprm_corr['stage_spearman']) > 0.6]
            for idx in df_filtr.index:
                plt.figure(figsize=(7, 5))
                sns_plot = sns.scatterplot(data=data, x = "stage", y = idx) #.set(title = 'pearson_' + str(df_filtr.loc[idx].values[0]))
                fig = sns_plot.get_figure()
                plt.title('pearson ' + str(round(prsn_corr.loc[idx].values[0],3)) + ' spearman ' + str(round(sprm_corr.loc[idx].values[0],3)))
                fig.savefig(os.path.join(output_dir, 'correlation', idx + "_scatterplot.png"))

                plt.figure(figsize=(7, 5))
                sns_plot = sns.boxplot(data=data, x="stage",y=idx)  # .set(title = 'pearson_' + str(df_filtr.loc[idx].values[0]))
                fig = sns_plot.get_figure()
                plt.title('pearson ' + str(round(prsn_corr.loc[idx].values[0], 3)) + ' spearman ' + str(round(sprm_corr.loc[idx].values[0],3)))
                fig.savefig(os.path.join(output_dir, 'correlation', idx + "_boxplot.png"))

        df_pearson = pd.concat(df_pearson)
        df_spearman = pd.concat(df_spearman)
        df = df_spearman.merge(df_pearson, left_index=True, right_index=True).round(3)
        df.to_csv(os.path.join(output_dir, 'correlation', mode + "_correlation.csv"))
        pd.concat(df_hand_face).to_csv(os.path.join(output_dir, 'correlation', mode + "_UPDRS_hand_face_mean_correlation.csv"))



    def processing(self, output_dir):
        for mode in self.config['mode']:
            if self.config['save_diagram']:
                data = pd.read_csv(os.path.join(self.config['path'], self.config['folder'], self.config[mode]['file']))
                data['stage'] = data['stage'].replace({2.5: self.config[2.5], 3.5: self.config[3.5]})  # TODO
                #data = data[data['dataset'].isin(self.config['datasets'])]
                data = data[data['dataset'].isin(self.config[mode]['datasets'])]
                data = data[data['face data quality'].isin(self.config['face_data_quality'])]
                data = data[data['hand data quality'].isin(self.config['hand_data_quality'])]
                ids = []
                for dataset in self.config[mode]['datasets']:
                    ids.extend([self.config[dataset]['id_name']+str(number) for number in self.config[dataset]['number']])
                data = data[data['id'].isin(ids)]
                self.save_plots(data, output_dir, mode)
                self.correlation_matrix(data, output_dir, mode)

        data = pd.read_csv(os.path.join(self.config['path'], self.config['folder'], 'feature_dataset.csv'))
        data['stage'] = data['stage'].replace({2.5: self.config[2.5], 3.5: self.config[3.5]})
        data = data[data['face data quality'].isin(self.config['face_data_quality'])]
        data = data[data['hand data quality'].isin(self.config['hand_data_quality'])]
        ids = []
        for dataset in self.config['datasets']:
            ids.extend([self.config[dataset]['id_name'] + str(number) for number in self.config[dataset]['number']])
        data = data[data['id'].isin(ids)]
        self.calculate_meta_statistic(data, output_dir)

