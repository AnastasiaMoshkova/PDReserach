import pandas as pd
import numpy as np
import os
import json
import math
from functools import reduce
from hydra.utils import instantiate


class ErrorAutoMarking():
    def __init__(self, config):
        self.config = config

    def _calc_error(self, path_to_manual, path_to_auto, datasets, mode):
        df_manual = pd.read_csv(path_to_manual)
        df_auto = pd.read_csv(path_to_auto)
        feature = self.config[mode]['feature']
        df_manual = df_manual[df_manual['dataset'].isin(datasets)] #[feature]
        if mode == 'hand':
            df_manual['key_id'] = df_manual['id'] + '_' + df_manual['r'].apply(str) + '_'+ df_manual['m'] + '_' + df_manual['hand']
            df_auto['key_id'] = df_auto['id'] + '_' + df_auto['r'].apply(str) + '_' + df_auto['m'] + '_' + df_auto['hand']
        if mode == 'face':
            df_manual['key_id'] = df_manual['id'] + '_' + df_manual['r'].apply(str)
            df_auto['key_id'] = df_auto['id'] + '_' + df_auto['r'].apply(str)
        result_relative = []
        result_mae = []
        for key_id in df_manual['key_id']:
            dataset = df_manual[df_manual['key_id'] == key_id]['dataset']
            feature_manual = df_manual[df_manual['key_id'] == key_id][feature]
            feature_auto = df_auto[df_auto['key_id'] == key_id][feature]
            if len(feature_auto) != 0:
                relative = abs((feature_auto - feature_manual)/feature_manual)*100
                relative['key_id'] = key_id
                relative['dataset'] = dataset
                mae = abs(feature_auto - feature_manual)
                mae['key_id'] = key_id
                mae['dataset'] = dataset
                result_relative.append(relative.round(self.config['round']))
                result_mae.append(mae.round(self.config['round']))
            else:
                result_relative.append(pd.DataFrame({'key_id':key_id}, index = [0]))
                result_mae.append(pd.DataFrame({'key_id':key_id}, index = [0]))
        df_r = pd.concat(result_relative)
        df_m = pd.concat(result_mae)
        #df = df_auto.merge(df_manual, left_on= ['id', 'r', 'dataset', 'm', 'hand'], right_on= ['id', 'r', 'dataset', 'm', 'hand'])
        return df_r.describe().loc['mean'], df_r.describe().loc['std'], df_m.describe().loc['mean'], df_m.describe().loc['std'], df_r, df_m

    def processing(self, output_dir):
        for mode in self.config['mode']:
            result = []
            path_to_manual = os.path.join(self.config['path'], self.config['folder_mannual'], self.config[mode]['path_to_manual'])
            path_to_auto = os.path.join(self.config['path'], self.config['folder_auto'], self.config[mode]['path_to_auto'])
            mean_r, std_r, mean_m, std_m, df_r, df_m = self._calc_error(path_to_manual, path_to_auto, self.config['dataset_type'], mode)
            result.append(pd.DataFrame(mean_r).rename(columns={'mean':'relative_error_mean'}).transpose())
            result.append(pd.DataFrame(std_r).rename(columns={'std':'relative_error_std'}).transpose())
            result.append(pd.DataFrame(mean_m).rename(columns={'mean':'mae_error_mean'}).transpose())
            result.append(pd.DataFrame(std_m).rename(columns={'std':'mae_error_std'}).transpose())
            df_r.to_csv(os.path.join(output_dir, mode + "_relative_error_by_keyid.csv"))
            df_m.to_csv(os.path.join(output_dir, mode + "_mae_error_by_keyid.csv"))
            for dataset in self.config['dataset_type']:
                mean_r, std_r, mean_m, std_m, df_r, df_m = self._calc_error(path_to_manual, path_to_auto, [dataset], mode)
                result.append(pd.DataFrame(mean_r).rename(columns={'mean':dataset+'_relative_error_mean'}).transpose())
                result.append(pd.DataFrame(std_r).rename(columns={'std':dataset+'_relative_error_std'}).transpose())
                result.append(pd.DataFrame(mean_m).rename(columns={'mean':dataset+'_mae_error_mean'}).transpose())
                result.append(pd.DataFrame(std_m).rename(columns={'std':dataset+'_mae_error_std'}).transpose())
            pd.concat(result).to_csv(os.path.join(output_dir, mode + "_error.csv"))




