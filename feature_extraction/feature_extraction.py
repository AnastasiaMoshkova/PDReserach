import pandas as pd
import numpy as np
import os
import json
import math
import glob
from functools import reduce
from hydra.utils import instantiate
from data_base.tremor import TremorProcessing


class FE():
    def __init__(self, config):
        self.config_fe = config['feature_extraction']
        self.config_feature = config['feature']
        self.TR = TremorProcessing()

    def dataset_processing(self):
        print('dataset_processing')
        dfs = []
        for dataset in self.config_fe['meta_data']['dataset_type']:
            folder_path = []
            r_number = []
            number = []
            df = pd.read_csv(self.config_fe[dataset]['path_to_meta'])
            df = df[['№', 'ФИО', 'Год рождения', 'Стадия по Хен-Яру', 'Возраст', 'Пол', 'face data quality', 'hand data quality', 'Сумма баллов ч.1, 2 и 4', 'Сумма баллов (только моторика, часть 3 UPDRS)', 'Сумма баллов (все части UPDRS)', 'UPDRS_HAND_FACE', '3.2_Выразительность лица', '3.4a_FT','3.4b_FT',	'3.5a_OC', '3.5b_OC', '3.6a_PS', '3.6b_PS']] #.drop([0,1])
            df['Стадия по Хен-Яру'] = df['Стадия по Хен-Яру'].replace('Атипичный', -1).replace('2,5', 2.5).replace('3,5', 3.5).apply(float)
            df[['Сумма баллов ч.1, 2 и 4', 'Сумма баллов (только моторика, часть 3 UPDRS)', 'Сумма баллов (все части UPDRS)', 'UPDRS_HAND_FACE', '3.2_Выразительность лица', '3.4a_FT','3.4b_FT',	'3.5a_OC', '3.5b_OC', '3.6a_PS', '3.6b_PS']] = df[['Сумма баллов ч.1, 2 и 4', 'Сумма баллов (только моторика, часть 3 UPDRS)', 'Сумма баллов (все части UPDRS)', 'UPDRS_HAND_FACE', '3.2_Выразительность лица', '3.4a_FT','3.4b_FT','3.5a_OC', '3.5b_OC', '3.6a_PS', '3.6b_PS']].replace(regex={',': '.'}).astype(float)
            df['№'] = df['№'].apply(int)
            folders = os.listdir(self.config_fe[dataset]['path_to_directory'])
            for folder in folders:
                folders_r = os.listdir(os.path.join(self.config_fe[dataset]['path_to_directory'],folder))
                for r in folders_r:
                    folder_path.append(os.path.join(self.config_fe[dataset]['path_to_directory'], folder, r))
                    r_number.append(int(r.split('r')[1]))
                    number.append(int(folder.split(self.config_fe[dataset]['id_name'])[1]))
            df_meta = pd.DataFrame()
            df_meta['№'] = number
            df_meta['r'] = r_number
            df_meta['path_to_folder'] = folder_path
            df_meta['dataset'] = dataset
            df_meta['id'] = self.config_fe[dataset]['id_name'] + df_meta['№'].apply(str)
            df = df.merge(df_meta, left_on='№', right_on='№')
            df = df.loc[df['№'].isin(self.config_fe[dataset]['number'])]
            df = df.rename(columns={'Пол': 'gender', 'Возраст': 'age', 'Стадия по Хен-Яру': 'stage', 'Год рождения': 'year', 'Сумма баллов ч.1, 2 и 4': 'UPDRS_1_2_4', 'Сумма баллов (только моторика, часть 3 UPDRS)': 'UPDRS_3', 'Сумма баллов (все части UPDRS)': 'UPDRS', '3.2_Выразительность лица':'UPDRS_mimic'})
            dfs.append(df)
            print('---{dataset} dataset processing finished---'.format(dataset=dataset))

        df = pd.concat(dfs, axis=0)
        df = df[df['stage'].isin(self.config_fe['meta_data']['stage'])]
        print('PD data:', df[df['dataset']=='PD'].shape[0])
        print('HC data:', df[df['dataset']=='HEALTHY'].shape[0])
        print('ST data:', df[df['dataset']=='STUDENT'].shape[0])
        return df

    '''
    def dataset_processing1(self):
        #TODO class for every dataset
        print('dataset_processing')
        for dataset in self.config_fe['meta_data']['dataset_type']:
            if dataset=='PD':
                df_pd = pd.read_csv(self.config_fe['pd_data']['path_to_meta'])
                df_pd = df_pd[['№', 'ФИО', 'Год рождения', 'Стадия по Хен-Яру', 'Возраст', 'Пол']].drop([0,1])
                df_pd['Стадия по Хен-Яру'] = df_pd['Стадия по Хен-Яру'].replace('Атипичный', -1)
                df_pd['Стадия по Хен-Яру'] = df_pd['Стадия по Хен-Яру'].replace('2,5', 2.5)
                df_pd['Стадия по Хен-Яру'] = df_pd['Стадия по Хен-Яру'].replace('3,5', 3.5)
                df_pd['Стадия по Хен-Яру'] = df_pd['Стадия по Хен-Яру'].apply(float)
                df_pd['№'] = df_pd['№'].apply(int)
                df_pd['dataset'] = 'PD'
                df_pd['path_to_folder'] = self.config_fe['pd_data']['path_to_directory']+'//Patient' + df_pd['№'].apply(str)
                df_pd['r'] = 0
                df_pd = df_pd.loc[df_pd['№'].isin(self.config_fe['pd_data']['pd_number'])]
                df_pd['id'] = 'Patient' + df_pd['№'].apply(str)
                df_pd = df_pd.rename(columns={'Пол': 'Gender', 'Возраст': 'Age'})
                print('---PD dataset processing finished---')
                #print(df_pd)

            if dataset == 'Healthy':
                df_healthy = pd.read_csv(self.config_fe['healthy_data']['path_to_meta'])
                healthy_r = []
                healthy_number = []
                r_number = []
                folders = os.listdir(self.config_fe['healthy_data']['path_to_directory'])
                for folder in folders:
                    folders_r = os.listdir(os.path.join(self.config_fe['healthy_data']['path_to_directory'],folder))
                    for r in folders_r:
                        healthy_r.append(os.path.join(self.config_fe['healthy_data']['path_to_directory'], folder, r))
                        r_number.append(int(r.split('r')[1]))
                        healthy_number.append(int(folder.split('Healthy')[1]))

                df_hc = pd.DataFrame(columns = ['№','dataset','r','path_to_folder','Стадия по Хен-Яру'])
                df_hc['№'] = healthy_number
                df_hc['r'] = r_number
                df_hc['path_to_folder'] = healthy_r
                df_hc['Стадия по Хен-Яру'] = 0
                df_hc['dataset'] = 'HEALTHY'
                df_hc['id'] = 'Healthy'+ df_hc['№'].apply(str)
                df_hc = df_healthy.merge(df_hc, left_on='Number', right_on='№')
                df_hc = df_hc.loc[df_hc['№'].isin(self.config_fe['healthy_data']['healthy_number'])]
                print('---Healthy dataset processing finished---')
                #print(df_hc)

            if dataset == 'Students':
                df_sudent = pd.read_csv(self.config_fe['students_data']['path_to_meta'])
                student_number = []
                student_folders = []
                folders = os.listdir(self.config_fe['students_data']['path_to_directory'])
                for folder in folders:
                    student_folders.append(os.path.join(self.config_fe['students_data']['path_to_directory'], folder))
                    student_number.append(int(folder.split('student')[1]))

                df_st = pd.DataFrame(columns=['№', 'dataset', 'r', 'path_to_folder', 'Стадия по Хен-Яру'])
                df_st['№'] = student_number
                df_st['r'] = 0
                df_st['path_to_folder'] = student_folders
                df_st['Стадия по Хен-Яру'] = 0
                df_st['dataset'] = 'STUDENT'
                df_st['id'] = 'Student' + df_st['№'].apply(str)
                df_st = df_sudent.merge(df_st, left_on='Number', right_on='№')
                df_st = df_st.loc[df_st['№'].isin(self.config_fe['students_data']['students_number'])]
                print('---Students dataset processing finished---')
                #print(df_st)

        df = pd.concat([df_pd, df_hc, df_st], axis=0)
        df = df.rename(columns={'Стадия по Хен-Яру': 'stage', 'Год рождения': 'year'})
        df = df[df['stage'].isin(self.config_fe['meta_data']['stage'])]
        print('PD data:', df[df['dataset']=='PD'].shape[0])
        print('HC data:', df[df['dataset']=='HEALTHY'].shape[0])
        print('ST data:', df[df['dataset']=='STUDENT'].shape[0])
        return df
    '''
    def loadfileInterval_hand(self, datapoint, start, stop):
        counter = 0
        maxPointX = []
        minPointX = []
        maxPointY = []
        minPointY = []
        newlist = []
        newlistSortedAll = sorted(datapoint, key=lambda k: k['X'])
        for point in newlistSortedAll:
            if ((point['X'] >= start) & (point['X'] <= stop)):
                newlist.append(point)
        for point in newlist:
            counter = counter + 1
            if point['Type'] == 1:
                if ((counter != 1) and (counter != len(newlist))):
                    maxPointX.append(point['X'])
                    maxPointY.append(point['Y'])
            if point['Type'] == 0:
                minPointX.append(point['X'])
                minPointY.append(point['Y'])
        return maxPointX, maxPointY, minPointX, minPointY

    def deleterAmplitude(self, maxPointX, maxPointY, minPointX, minPointY, threshhold):
        resultMax = []
        resultMin = []
        for i in range(len(maxPointX)):
            if (maxPointY[i] - minPointY[i + 1]) < threshhold:
                resultMax.append(i)
                resultMin.append(i + 1)

        if len(resultMax) != 0:
            resultMax.reverse()
            for k in resultMax:
                del maxPointX[k]
                del maxPointY[k]
        if len(resultMin) != 0:
            resultMin.reverse()
            for k in resultMin:
                del minPointX[k]
                del minPointY[k]
        return maxPointX, maxPointY, minPointX, minPointY

    def loadfile_face(self, filepoint, au, frame_rate):
        maxPointX = []
        minPointX = []
        maxPointY = []
        minPointY = []
        key = [k for k in filepoint.keys() if au in k]
        if len(key)!=0:
            datapoint = filepoint[key[0]]
            counter = 0
            newlist = sorted(datapoint, key=lambda k: k['X'])

            if len(newlist) == 0:
                return maxPointX, maxPointY, minPointX, minPointY

            for point in newlist:
                counter = counter + 1
                if point['Type'] == 1:
                    if ((counter != 1) and (counter != len(newlist))):
                        maxPointX.append(point['X'] / frame_rate)
                        maxPointY.append(point['Y'])
                if point['Type'] == 0:
                    minPointX.append(point['X'] / frame_rate)
                    minPointY.append(point['Y'])
            if ((len(maxPointY) - len(minPointY)) != -1):
                print("########################################")
            if (len(maxPointY) == 0):
                print("NUUUUL")

            number_movements = self.config_fe['feature_extractor']['face']['number_movements']
            if (len(maxPointX) > number_movements):
                return maxPointX[0:number_movements], maxPointY[0:number_movements], minPointX[0:number_movements+1], minPointY[0:number_movements+1]
            else:
                return maxPointX, maxPointY, minPointX, minPointY
        else:
            return maxPointX, maxPointY, minPointX, minPointY


    def feature1(self, data):
        return len(data)

    def image_aus(self, folder, aus, loc):
        files_em = os.listdir(folder)
        image_df_em = []
        if loc:
            files = glob.glob(folder+"//*.csv")
            files = [{'n': int(file.split('\\')[-1].split('_')[-1].split('.csv')[0]), 'file': file} for file in files]
            sorted_files = sorted(files, key=lambda k: k['n'])
            files = [file['file'] for file in sorted_files]
            n = int(len(files) / 2)
            files_em = files[n - 3:n + 3]
        for file in files_em:
            if '.csv' in file:
                df = pd.read_csv(os.path.join(folder, file))
                for au in aus:
                    df[au] = df[' ' + au + '_r'] * df[' ' + au + '_c']
                image_df_em.append(df)
        df_em = pd.concat(image_df_em, axis=0)
        df_em = df_em[df_em[' confidence'] > 0.8]
        return df_em

    def feature_em(self,folder_em,folder_p1,aggregation):
        aus = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14','AU15','AU17', 'AU20','AU23', 'AU25', 'AU26'] #, 'AU45']
        df_em = self.image_aus(folder_em, aus, self.config_fe['feature_extractor']['em']['median_image'])
        df_p1 = self.image_aus(folder_p1, aus, False)
        df_p1_agg = df_p1[aus].describe().loc[aggregation]
        feature = []
        for i in range(len(df_em)):
            dist = []
            for au in aus:
                dist.append((df_em.iloc[i][au] - df_p1_agg[au])**2)
            feature.append(math.sqrt(np.sum(dist)))
        if self.config_fe['feature_extractor']['em']['frame_au_agg'] == 'max':
            return np.max(feature)
        if self.config_fe['feature_extractor']['em']['frame_au_agg'] == 'mean':
            return np.mean(feature) #np.max(feature) #np.mean(feature) #TODO different

    def feature_difference_between_hand(self, df):
        features = [name + '_' + ex for ex in self.config_fe['feature_extractor']['hand']['exercise'] for name in self.config_fe['feature_extractor']['hand']['feature_type']]
        agg = {'id': 'first', 'r': 'first', 'm': 'first'}
        f = lambda x: abs(x.diff().dropna().values[0]) if ((len(x.dropna()) == 2) & (-1 not in set(x))) else np.NaN
        for feature in features:
            agg.update({feature: f})
        features2 = ['id', 'r', 'm']
        features2.extend(features)
        df2 = df.groupby(['id', 'r', 'm'], as_index=False)[features2].agg(agg).rename(columns=dict(zip(features, [fe.split('_')[0] + 'Diff_' + fe.split('_')[1] for fe in features])))
        return df.merge(df2, left_on=['id', 'r', 'm'], right_on=['id', 'r', 'm'])

    def feature_calculation_hand(self, path, file, exercise):
        hand = file.split('_')[1]
        m = file.split('_')[2]
        patient_id = file.split('_')[3]
        path_to_file = os.path.join(path, file)
        datapoint = json.load(open(path_to_file))
        start = self.config_fe['feature_extractor']['hand']['start']
        stop = self.config_fe['feature_extractor']['hand']['stop']
        maxPointX, maxPointY, minPointX, minPointY = self.loadfileInterval_hand(datapoint, start, stop)
        algorithm_filtering = self.config_fe['feature_extractor']['hand']['algorithm_filtering']
        if algorithm_filtering =='by_low_amplitude':
            maxPointX, maxPointY, minPointX, minPointY = self.deleterAmplitude(maxPointX, maxPointY, minPointX, minPointY, self.config_fe['feature_extractor']['hand']['threshold_aplitude'])
        norm_coeff = self.config_fe['feature_extractor']['hand']['norm_coeff']
        feature_type = self.config_fe['feature_extractor']['hand']['feature_type']
        d = {'m': m, 'hand': hand}
        for feature in feature_type:
            if len(maxPointX) > 1:
                #d.update({feature+'_'+exercise:self.feature1(data)})
                feature_class = instantiate(self.config_feature[feature], maxPointX, maxPointY, minPointX, minPointY, norm_coeff, datapoint)
                d.update({feature + '_' + exercise: feature_class.calc()})
            else:
                d.update({feature + '_' + exercise: -1})
        return d

    def feature_calculation_face(self, path, file, exercise):
        d = {}
        path_to_file = os.path.join(path, file,)
        feature_type = self.config_fe['feature_extractor']['face']['feature_type']
        if os.path.isfile(path_to_file):
            datapoint = json.load(open(path_to_file))
            algorithm_filtering = self.config_fe['feature_extractor']['face']['algorithm_filtering']
            for au in self.config_fe['feature_extractor']['face']['exercise'][exercise]:
                for feature in feature_type:
                    maxPointX, maxPointY, minPointX, minPointY = self.loadfile_face(datapoint, au, self.config_fe['feature_extractor']['face']['frame_rate'])
                    if algorithm_filtering == 'by_low_amplitude':
                        threshold_aplitude = self.config_fe['feature_extractor']['face']['threshold_aplitude']
                        maxPointX, maxPointY, minPointX, minPointY = self.deleterAmplitude(maxPointX, maxPointY, minPointX, minPointY,threshold_aplitude)
                    if len(maxPointX) > 2:
                        feature_class = instantiate(self.config_feature[feature], maxPointX, maxPointY, minPointX, minPointY, 1, datapoint)
                        #print(path_to_file)
                        #print(feature_class.calc())
                        d.update({feature + '_' + au + '_' + exercise: feature_class.calc()})
                    else:
                        d.update({feature + '_' + au + '_' + exercise: -1})
                    #d.update({feature + '_' + au + '_' + exercise: self.feature1(datapoint)})
        else:
            for au in self.config_fe['feature_extractor']['face']['exercise'][exercise]:
                for feature in feature_type:
                    d.update({feature + '_' + au + '_' + exercise: np.NaN})
        return d

    def feature_calculation_em(self, path_to_folder, exercise):
        d = {}
        folder_p1 = os.path.join(path_to_folder,'p1')
        aggregation = self.config_fe['feature_extractor']['em']['frame_number']
        if os.path.isdir(folder_p1):
            feature_type = self.config_fe['feature_extractor']['em']['feature_type']
            ex = {'DA':'p12', 'DI':'p13'}
            ems = {'Anger':'1', 'Disgust':'2', 'Fear':'3', 'Happiness':'4', 'Sadness':'5', 'Surprise':'6'}
            for feature in feature_type:
                folder_em = os.path.join(path_to_folder, ex[exercise]+'_'+ems[feature])
                if os.path.isdir(folder_em):
                    d.update({exercise  + '_' + feature: self.feature_em(folder_em,folder_p1,aggregation)})
                else:
                    d.update({exercise + '_' + feature: np.NaN})
        return d

    def feature_calculation_tremor(self, path_to_tr_folder, file, key_point):
        feature_type = self.config_fe['feature_extractor']['tremor']['feature_type']
        hand = file.split('_')[1]
        m = file.split('_')[2]
        if hand == 'RL':
            for hand_type in ['R', 'L']:
                d = {'m': m, 'hand': hand_type}
                tremor_features = self.TR.get_features(path_to_tr_folder, file, hand_type, key_point)
                for feature in feature_type:
                    if feature in tremor_features.keys():
                        d.update({feature + '_' + key_point: tremor_features[feature]})
                    else:
                        d.update({feature + '_' + key_point: np.NaN})
        else:
            d = {'m': m, 'hand': hand}
            tremor_features = self.TR.get_features(path_to_tr_folder, file, hand, key_point)
            for feature in feature_type:
                if feature in tremor_features.keys():
                    d.update({feature + '_' + key_point: tremor_features[feature]})
                else:
                    d.update({feature + '_' + key_point: np.NaN})
        return d

    def feature_extraction_dataset(self, df: pd.DataFrame):
        data = []
        data_face = []
        data_hand = []
        data_name = []
        columns = ['id', 'r', 'dataset', 'age', 'gender', 'stage', 'year', 'UPDRS_3', 'UPDRS_1_2_4', 'UPDRS','UPDRS_HAND_FACE', 'UPDRS_mimic', '3.4a_FT','3.4b_FT','3.5a_OC', '3.5b_OC', '3.6a_PS', '3.6b_PS', 'face data quality', 'hand data quality', 'path_to_folder']
        for mode in self.config_fe['meta_data']['feature_mode']:
            if mode=='hand':
                dfs = []
                for i in range(len(df)):
                    dfs.append(self.hand_feature_extraction_by_folder(df.iloc[i][columns]))
                df_hand = pd.concat(dfs, axis=0).reset_index()
                data_hand.append(df_hand)
                data.append(df_hand)
                data_name.append('hand')
                #df_hand = df.merge(df_hand, left_on=['id','r'], right_on=['id','r'], how = 'outer')
                #print(df)
                print('---Hand feature calculated---')
            if mode == 'face':
                dfs = []
                for i in range(len(df)):
                    dfs.append(self.face_feature_extraction_by_folder(df.iloc[i][columns]))
                df_face = pd.concat(dfs, axis=0).reset_index()
                data_face.append(df_face)
                data.append(df_face)
                data_name.append('face')
                #df_face = df.merge(df_face, left_on=['id', 'r'], right_on=['id', 'r'], how='outer')
                #print(df_face)
                print('---Face feature calculated---')
            if mode == 'em':
                dfs = []
                for i in range(len(df)):
                    dfs.append(self.em_feature_extraction_by_folder(df.iloc[i][columns]))
                df_em = pd.concat(dfs, axis=0).reset_index()
                data_face.append(df_em)
                data.append(df_em)
                data_name.append('em')
                #df_face = df.merge(df_face, left_on=['id', 'r'], right_on=['id', 'r'], how='outer')
                #print(df_face)
                print('---Em feature calculated---')
            if mode == 'tremor':
                dfs = []
                for i in range(len(df)):
                    dfs.append(self.tremor_feature_extraction_by_folder(df.iloc[i][columns]))
                df_tremor = pd.concat(dfs, axis=0).reset_index()
                data_hand.append(df_tremor)
                data.append(df_tremor)
                data_name.append('tremor')
                # df_face = df.merge(df_face, left_on=['id', 'r'], right_on=['id', 'r'], how='outer')
                # print(df_face)
                print('---Tremor feature calculated---')
        '''     
        df = df.merge(df_hand, left_on=['id', 'r'], right_on=['id', 'r'], how='outer')
        df = df.merge(df_face, left_on=['id', 'r'], right_on=['id', 'r'], how='outer')
        df = df.merge(df_em, left_on=['id', 'r'], right_on=['id', 'r'], how='outer')
        '''
        merged_columns = ['m', 'hand']
        merged_columns.extend(columns)
        df_merged_hand = reduce(lambda left, right: pd.merge(left, right, on=merged_columns, how='outer'), data_hand)
        df_merged = []
        df_merged.extend([df_merged_hand])
        df_merged.extend(data_face)
        df = reduce(lambda left, right: pd.merge(left, right, on=columns, how='outer'), df_merged)
        print(df)
        return df, data, data_name

    def processing(self, output_dir):
        df = self.dataset_processing()
        df, data, data_name = self.feature_extraction_dataset(df)
        saving_df_path = os.path.join(output_dir, "feature_dataset.csv")
        df.to_csv(saving_df_path)
        for i in range(len(data)):
            data[i].to_csv(os.path.join(output_dir, data_name[i] + "_feature.csv"))
    def hand_feature_extraction_by_folder(self, data):
        # TODO class for every feature extractor
        dfs = []
        path_to_point_folder = os.path.join(data['path_to_folder'], self.config_fe['feature_extractor']['hand']['path_to_point_folder'])
        if os.path.isdir(path_to_point_folder):
            for file in os.listdir(path_to_point_folder):
                ex = {'1': 'FT', '2': 'OC', '3': 'PS'}
                exercise_number = file.split('leapRecording')[1].split('_')[0]
                exercise = ex[exercise_number]
                if exercise in self.config_fe['feature_extractor']['hand']['exercise']:
                    dfs.append(self.feature_calculation_hand(path_to_point_folder, file, exercise))
            dfs = pd.DataFrame(dfs).groupby(['m','hand'], as_index=False).mean(numeric_only=True)
        else:
            print('No folder:', path_to_point_folder)
            dfs = pd.DataFrame()
        for col in data.index:
            dfs[col] = data[col]
        # dfs['id'] = id
        # dfs['r'] = r
        # dfs['dataset'] = dataset
        # dfs['age'] = age
        # dfs['gender'] = gender
        # dfs['stage'] = stage
        #calculate features between hands
        dfs = self.feature_difference_between_hand(dfs)

        return dfs

    def face_feature_extraction_by_folder(self, data):
        # TODO class for every feature extractor
        dfs = []
        path_to_point_folder = os.path.join(data['path_to_folder'], self.config_fe['feature_extractor']['face']['path_to_point_folder'])
        if os.path.isdir(path_to_point_folder):
            for file in os.listdir(path_to_point_folder):
                exercise = file.split('_')[0]
                exercises = list(self.config_fe['feature_extractor']['face']['exercise'].keys())
                if exercise in exercises:
                    dfs.append(self.feature_calculation_face(path_to_point_folder, file, exercise))
            dfs = pd.DataFrame(dfs)
        else:
            dfs = pd.DataFrame()
            print('No folder:', path_to_point_folder)
        dfs['id'] = data['id']
        dfs = pd.DataFrame(dfs).groupby(['id']).mean(numeric_only=True)
        for col in data.index:
            if col!='id':
                dfs[col] = data[col]
        # dfs['dataset'] = dataset
        # dfs['age'] = age
        # dfs['gender'] = gender
        # dfs['stage'] = stage
        return dfs

    def em_feature_extraction_by_folder(self, data):
        # TODO class for every feature extractor
        dfs = []
        for exercise in self.config_fe['feature_extractor']['em']['exercise']:
            path_to_folder = os.path.join(data['path_to_folder'], 'face')
            dfs.append(self.feature_calculation_em(path_to_folder, exercise))
        dfs = pd.DataFrame(dfs)
        dfs['id'] = data['id']
        dfs = pd.DataFrame(dfs).groupby(['id']).mean(numeric_only=True)
        for col in data.index:
            if col!='id':
                dfs[col] = data[col]
        # dfs['id'] = id
        # dfs['r'] = r
        # dfs = pd.DataFrame(dfs).groupby(['id']).mean(numeric_only=True)
        # dfs['dataset'] = dataset
        # dfs['age'] = age
        # dfs['gender'] = gender
        # dfs['stage'] = stage
        return dfs

    def tremor_feature_extraction_by_folder(self, data):
        # TODO class for every feature extractor
        dfs = []
        path_to_tr_folder = os.path.join(data['path_to_folder'], 'hand')
        if os.path.isdir(path_to_tr_folder):
            for file in os.listdir(path_to_tr_folder):
                for key_point in self.config_fe['feature_extractor']['tremor']['key_point']:
                    if (('TR' in file) & ('.json' in file)):
                        dfs.append(self.feature_calculation_tremor(path_to_tr_folder, file, key_point))
            if len(dfs) != 0:
                dfs = pd.DataFrame(dfs).groupby(['m','hand'], as_index=False).mean(numeric_only=True)
                # dfs = self.feature_difference_between_hand(dfs)
                for col in data.index:
                    dfs[col] = data[col]
            else:
                print('No files with TR:', path_to_tr_folder)
                dfs = pd.DataFrame()
        else:
            print('No folder:', path_to_tr_folder)
            dfs = pd.DataFrame()
        return dfs

    '''
    def DeleterPoint1(point_sort):
        deleter = []
        counter = 0
        print(len(point_sort))
        for i in range(len(point_sort)):
            counter = counter + point_sort[i]['type']
            if (counter == 2):
                if ([point_sort[i]['value'] > point_sort[i - 1]['value']]):
                    deleter.append(i - 1)
                    counter = 1
                else:
                    deleter.append(i)
                    counter = 1
            if (counter == -2):
                if ([point_sort[i]['value'] < point_sort[i - 1]['value']]):
                    deleter.append(i - 1)
                    counter = -1
                else:
                    deleter.append(i)
                    counter = -1

        deleter.reverse()
        for i in range(len(deleter)):
            del point_sort[deleter[i]]

        return point_sort

    def DeleterPoint2(point_sort, level):
        deleter = []
        for i in range(len(point_sort) - 1):
            if ((abs(point_sort[i + 1]['value'] - point_sort[i]['value']) < level)):
                print(i)
                deleter.append(i)

            deleter2 = list(set(deleter))
            print(deleter2)
            deleter2.reverse()
            print(deleter2)
        for i in range(len(deleter2)):
            del point_sort[deleter2[i]]

        return point_sort
    '''













