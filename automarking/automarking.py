import os
import pandas as pd
import json
import math
import numpy as np
import json
import os
from hydra.utils import instantiate
import shutil
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from data_base.hand3D import HandData
from data_base.hand2D import HandDataAngle

'''
реализация класса расстановки точек экстремума с учетом выбранной модальности
'''

class AutoMarking():
    def __init__(self, config):
        self.config = config

    #подготовка датафрема с путями до папок, где будет применен алгоритм авто разметки
    def data_parser_dataset(self, dataset):
        path_to_dir = self.config['automarking'][dataset]['path_to_directory']
        folders = os.listdir(self.config['automarking'][dataset]['path_to_directory'])
        id = []
        dataset_folders = []
        for folder in folders:
            folders_r = os.listdir(os.path.join(path_to_dir, folder))
            for r in folders_r:
                id.append(folder)
                dataset_folders.append(os.path.join(path_to_dir, folder, r))
        df = pd.DataFrame(dataset_folders, columns=['path'])
        df['folders'] = id
        folder_name = self.config['automarking'][dataset]['folder_name']
        df['number'] = df['folders'].str.split(folder_name).str[1].apply(int)
        df['dataset'] = dataset
        return df.loc[df['number'].isin(self.config['automarking'][dataset]['number'])]

    def data_processing(self):
        df_result = []
        for dataset in self.config['automarking']['dataset_type']:
            df_result.append(self.data_parser_dataset(dataset))
            '''
            if ((dataset == 'PD') | (dataset == 'Students')):
                #df_result.append(self.data_parser(dataset))
                df_result.append(self.data_parser_dataset(dataset))
            if (dataset == 'Healthy'):
                df_result.append(self.data_parser_dataset(dataset))
            '''
        df = pd.concat(df_result, axis = 0).reset_index(drop=True)
        return df

    #запись точек в файл
    def write_point_hand(self, path, file, maxP, minP, maxA, minA, frac, order_min, order_max):
        datapoint = []
        for i in range(len(maxP)):
            datapoint.append({"Type": 1, "Scale": 1.0, "Brush": "#FFFF0000", "X": float(maxP[i]), "Y": maxA[i]})
        for i in range(len(minP)):
            datapoint.append({"Type": 0, "Scale": 1.0, "Brush": "#FF0000FF", "X": float(minP[i]), "Y": minA[i]})
        datapoint = sorted(datapoint, key=lambda k: k['X'])
        file_point = file.split('.json')[0] + '_'.join(['_point', str(frac), str(order_min), str(order_max)])+'.json'
        if len(datapoint) != 0:
            if not os.path.isdir(path):
                os.mkdir(path)
            with open(os.path.join(path, file_point), 'w') as f:
                json.dump(datapoint, f)

    def write_point_hand_timestamps(self, path, file, maxP, minP, maxA, minA, frac, order_min, order_max):
        datapoint = []
        for i in range(len(maxP)):
            datapoint.append({"Type": 1, "Scale": 1.0, "Brush": "#FFFF0000", "X": float(maxP[i]), "Y": maxA[i]})
        for i in range(len(minP)):
            datapoint.append({"Type": 0, "Scale": 1.0, "Brush": "#FF0000FF", "X": float(minP[i]), "Y": minA[i]})
        datapoint = sorted(datapoint, key=lambda k: k['X'])
        file_point = file.split('.json')[0] + '_'.join(['_point', str(frac), str(order_min), str(order_max)])+'.json'
        if len(datapoint) != 0:
            if not os.path.isdir(path):
                os.mkdir(path)
            with open(os.path.join(path, file_point), 'w') as f:
                json.dump(datapoint, f)

    #запись точек в словарь
    def write_point_face(self, maxP, minP, maxA, minA):
        datapoint = []
        for i in range(len(maxP)):
            datapoint.append({"Type": 1, "Scale": 1.0, "Brush": "#FFFF0000", "X": float(maxP[i]), "Y": maxA[i]})
        for i in range(len(minP)):
            datapoint.append({"Type": 0, "Scale": 1.0, "Brush": "#FF0000FF", "X": float(minP[i]), "Y": minA[i]})
        datapoint = sorted(datapoint, key=lambda k: k['X'])
        return datapoint

    #запись точек в .json и сохранение в соответсвующую папку
    def save_point_face(self, path, file, datapoint):
        if not os.path.isdir(path):
            os.mkdir(path)
        with open(os.path.join(path, file.split('.')[0] + '.json'), 'w') as f:
            json.dump(datapoint, f)



    #получение сигналов мимической активности
    def signal_face(self, file, au):
        au_dict = {'AU1': 'AU01', 'AU2': 'AU02', 'AU3': 'AU03', 'AU4': 'AU04', 'AU5': 'AU05',
                   'AU6': 'AU06', 'AU7': 'AU07', 'AU8': 'AU08', 'AU9': 'AU09', 'AU12':'AU12', 'AU14':'AU14'}
        df = pd.read_csv(file)
        if len(df) != 0:
            values = df[' ' + au_dict[au] + '_c'] * df[' ' + au_dict[au] + '_r']
            frame = df['frame']
        else:
            values, frame = [], []
        return values, frame

    def signal_AU(self, file, au):
        pass

    #получение точек авто разметки по сигналам двигательной активности рук
    def auto_point_hand(self, values, frame, mode):
        auto_alg_class = instantiate(self.config['automarking'][mode]['auto_alg_class'])
        maxP, minP, maxA, minA, frac, order_min, order_max = auto_alg_class.get_point(values, frame)
        return np.array(frame)[maxP], np.array(frame)[minP], maxA, minA, frac, order_min, order_max
        #return maxP, minP, maxA, minA, frac, order_min, order_max

    #получение точек авто разметки по сигналам мимической активности
    def auto_point_face(self, values, frame):
        auto_alg_class = instantiate(self.config['automarking']['face']['auto_alg_class'])
        maxP, minP, maxA, minA, frac, order_min, order_max = auto_alg_class.get_point(values, frame)
        return maxP, minP, maxA, minA, frac, order_min, order_max


    def hand_processing_auto_point(self, path, output_dir, type_signal, mode):
        signal_class = instantiate(self.config['automarking']['signal_class'][type_signal])
        input_folder = self.config['automarking'][mode]['input_folder']
        exercise_dict = {'FT':'1', 'OC':'2', 'PS':'3'}
        exercises = [exercise_dict[ex] for ex in self.config['automarking'][mode]['exercise']]
        folder_to_save = os.path.join(path, self.config['automarking'][mode]['output_folder'] + '_' + type_signal)
        if os.path.isdir(folder_to_save):
            shutil.rmtree(folder_to_save)
        if os.path.isdir(os.path.join(path, input_folder)):
            for file in os.listdir(os.path.join(path, input_folder)):
                if '.json' in file:
                    exercise = file.split(self.config['automarking'][mode]['record_title'])[1].split('_')[0]
                    if exercise in exercises:
                        hand = file.split('_')[1]
                        values, frame, palm_width = signal_class.signal_hand(os.path.join(path, input_folder, file), exercise, hand)
                        if len(values) > self.config['automarking'][mode]['threshold_data_length']:
                            maxP, minP, maxA, minA, frac, order_min, order_max = self.auto_point_hand(values, frame, mode)
                            #maxP, minP, maxA, minA = self.signalPoint(maxP, minP, maxA, minA)
                            if self.config['automarking']['image_save']:
                                if not os.path.isdir(os.path.join(output_dir, mode)):
                                    os.mkdir(os.path.join(output_dir, mode))
                                path_to_save_image = os.path.join(output_dir, mode, file.split('.')[0] + '_' + mode + '_' + type_signal + '.png')
                                signal_class.plot_image(values, frame, maxP, minP, maxA, minA, path_to_save_image, file.split('.')[0])
                            self.write_point_hand(folder_to_save, file, maxP, minP, maxA, minA, frac, order_min, order_max)

    def face_processing_auto_point(self, path, output_dir):
        exercises = self.config['automarking']['face']['exercise'].keys()
        folder_to_save = os.path.join(path, self.config['automarking']['face']['output_folder'])
        if os.path.isdir(folder_to_save):
            shutil.rmtree(folder_to_save)
        if os.path.isdir(os.path.join(path,'face')):
            for file in os.listdir(os.path.join(path,'face')):
                if '.csv' in file:
                    exercise = file.split('_')[0]
                    if exercise in exercises:
                        aus = self.config['automarking']['face']['exercise'][exercise]
                        au_point_dict = {}
                        for au in aus:
                            values, frame = self.signal_face(os.path.join(path, 'face', file), au)
                            if len(values) > self.config['automarking']['face']['threshold_data_length']:
                                maxP, minP, maxA, minA, frac, order_min, order_max = self.auto_point_face(values, frame)
                                #maxP, minP, maxA, minA = self.signalPoint(maxP, minP, maxA, minA)
                                if self.config['automarking']['image_save']:
                                    if not os.path.isdir(os.path.join(output_dir, 'face')):
                                        os.mkdir(os.path.join(output_dir, 'face'))
                                    path_to_save_image = os.path.join(output_dir, 'face', file.split('.csv')[0] + '_' + au+'.png')
                                    self.plot_image(values, frame, maxP, minP, maxA, minA, path_to_save_image, file.split('.json')[0]+ '_'+ au)
                                datapoint = self.write_point_face( maxP, minP, maxA, minA)
                                au_point_dict.update({au:datapoint})
                        self.save_point_face(folder_to_save, file, au_point_dict)

    def processing(self, output_dir):
        df = self.data_processing()
        for path in df['path']:
            for mode in self.config['automarking']['mode']:
                if ((mode=='hand') | (mode=='handMP')):
                    for task_type in self.config['automarking']['signal_class'].keys():
                        self.hand_processing_auto_point(path, output_dir, task_type, mode)
                if mode=='face':
                    self.face_processing_auto_point(path, output_dir)
