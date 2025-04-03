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

    '''
    def data_parser(self, dataset):
        df_result = []
        path_to_dir = self.config['automarking'][dataset]['path_to_directory']
        folders = os.listdir(self.config['automarking'][dataset]['path_to_directory'])
        df = pd.DataFrame(folders, columns=['folders'])
        df['path'] = path_to_dir + '//' + df['folders']
        folder_name = self.config['automarking'][dataset]['folder_name']
        df['number'] = df['folders'].str.split(folder_name).str[1].apply(int)
        df['dataset'] = dataset
        df_result.append(df.loc[df['number'].isin(self.config['automarking'][dataset]['number'])])
        return pd.concat(df_result).reset_index(drop=True)
    '''
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
        with open(os.path.join(path, file), 'w') as f:
            json.dump(datapoint, f)

    def plot_image(self, values, frame, maxPointX, minPointX, maxPointY, minPointY, path_to_save, title):
        figure(figsize=(32, 6), dpi=80)
        plt.plot(frame, values)
        plt.plot(maxPointX, maxPointY, 'ro')
        plt.plot(minPointX, minPointY, 'bo')
        plt.xlabel('frame')
        plt.ylabel('Distance')
        plt.title(title)
        plt.savefig(path_to_save)
        plt.clf()

    #получение сигналов двигательной активности рук по номеру упражнения
    def signal_exersice_hand(self, data, hand, exersice):
        if exersice=='1':
            values, frame = self.signal_FT(data, hand)
        if exersice=='2':
            values, frame = self.signal_OC(data, hand)
        if exersice=='3':
            values, frame = self.signal_PS(data, hand)
        return values, frame

    #получение сигналов двигательной активности рук
    def signal_hand(self, file, exersice, hand_type):
        hand_dict = {'L':'left hand', 'R':'right hand'}
        hand = hand_dict[hand_type]
        data = json.load(open(file))
        values, frame = self.signal_exersice_hand(data, hand, exersice)
        if len(values) < self.config['automarking']['hand']['threshold_data_length']: #TODO #if len(values)==0:
            del hand_dict[hand_type]
            hand = hand_dict[list(hand_dict.keys())[0]]
            values, frame = self.signal_exersice_hand(data, hand, exersice)
        return values, frame

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

    #построение сигнала "постукивание пальцами"
    def signal_FT(self, data, hand):
        frame = []
        values = []
        for i in range(len(data)):
            if hand in data[i].keys():
                sum_sqr = ((float(data[i][hand]['FORE_TIP']['X1']) - float(data[i][hand]['THUMB_TIP']['X1'])) ** 2 +
                           (float(data[i][hand]['FORE_TIP']['Y1']) - float(data[i][hand]['THUMB_TIP']['Y1'])) ** 2 +
                           (float(data[i][hand]['FORE_TIP']['Z1']) - float(data[i][hand]['THUMB_TIP']['Z1'])) ** 2)
                distance = math.sqrt(sum_sqr)
                values.append(distance)
                frame.append(data[i][self.config['automarking']['hand']['timestamp']])
        return values, frame

    #построение сигнала "открытие/закрытие ладони"
    def signal_OC(self, data, hand):
        frame = []
        values = []
        for i in range(len(data)):
            if hand in data[i].keys():
                sum_sqr = (float(data[i][hand]['MIDDLE_TIP']['X1']) - float(data[i][hand]['CENTRE']['X'])) ** 2 + (
                                  float(data[i][hand]['MIDDLE_TIP']['Y1']) - float(data[i][hand]['CENTRE']['Y'])) ** 2 + (
                                      float(data[i][hand]['MIDDLE_TIP']['Z1']) - float(data[i][hand]['CENTRE']['Z'])) ** 2
                distance = math.sqrt(sum_sqr)
                values.append(distance)
                frame.append(data[i][self.config['automarking']['hand']['timestamp']])
        return values, frame

    # построение сигнала "пронация/супинация ладони"
    def signal_PS(self, data, hand):
        frame = []
        values = []
        for i in range(len(data)):
            if hand in data[i].keys():
                values.append(float(data[i][hand]['CENTRE']['Angle']))
                frame.append(data[i][self.config['automarking']['hand']['timestamp']])
        return values, frame

    def signal_AU(self, file, au):
        pass

    #получение точек авто разметки по сигналам двигательной активности рук
    def auto_point_hand(self, values, frame):
        auto_alg_class = instantiate(self.config['automarking']['hand']['auto_alg_class'])
        maxP, minP, maxA, minA, frac, order_min, order_max = auto_alg_class.get_point(values, frame)
        return maxP, minP, maxA, minA, frac, order_min, order_max

    #получение точек авто разметки по сигналам мимической активности
    def auto_point_face(self, values, frame):
        auto_alg_class = instantiate(self.config['automarking']['face']['auto_alg_class'])
        maxP, minP, maxA, minA, frac, order_min, order_max = auto_alg_class.get_point(values, frame)
        return maxP, minP, maxA, minA, frac, order_min, order_max


    def leap_processing_auto_point(self, path, output_dir):
        input_folder = self.config['automarking']['hand']['input_folder']
        exercise_dict = {'FT':'1', 'OC':'2', 'PS':'3'}
        exercises = [exercise_dict[ex] for ex in self.config['automarking']['hand']['exercise']]
        folder_to_save = os.path.join(path, self.config['automarking']['hand']['output_folder'])
        if os.path.isdir(folder_to_save):
            shutil.rmtree(folder_to_save)
        if os.path.isdir(os.path.join(path, input_folder)):
            for file in os.listdir(os.path.join(path, input_folder)):
                if '.json' in file:
                    exercise = file.split('leapRecording')[1].split('_')[0]
                    if exercise in exercises:
                        hand = file.split('_')[1]
                        values, frame = self.signal_hand(os.path.join(path, input_folder, file), exercise, hand)
                        if len(values) > self.config['automarking']['hand']['threshold_data_length']:
                            maxP, minP, maxA, minA, frac, order_min, order_max = self.auto_point_hand(values, frame)
                            #maxP, minP, maxA, minA = self.signalPoint(maxP, minP, maxA, minA)
                            if self.config['automarking']['image_save']:
                                if not os.path.isdir(os.path.join(output_dir, input_folder)):
                                    os.mkdir(os.path.join(output_dir, input_folder))
                                path_to_save_image = os.path.join(output_dir, input_folder, file.split('.json')[0]+'png')
                                self.plot_image(values, frame, maxP, minP, maxA, minA, path_to_save_image, file.split('.json')[0])
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



    def signal_FT(self, data, hand):
        frame = []
        values = []
        for i in range(len(data)):
            if hand in data[i].keys():
                sum_sqr = ((float(data[i][hand]['FORE_TIP']['X1']) - float(data[i][hand]['THUMB_TIP']['X1'])) ** 2 +
                           (float(data[i][hand]['FORE_TIP']['Y1']) - float(data[i][hand]['THUMB_TIP']['Y1'])) ** 2 +
                           (float(data[i][hand]['FORE_TIP']['Z1']) - float(data[i][hand]['THUMB_TIP']['Z1'])) ** 2)
                distance = math.sqrt(sum_sqr)
                values.append(distance)
                frame.append(data[i][self.config['automarking']['hand']['timestamp']])
        return values, frame


    

    def MP_processing_auto_point(self, path, output_dir):
        input_folder = self.config['automarking']['MP']['input_folder']
        exercise_dict = {'FT':'1', 'OC':'2', 'PS':'3'}
        exercises = [exercise_dict[ex] for ex in self.config['automarking']['MP']['exercise']]
        folder_to_save = os.path.join(path, self.config['automarking']['MP']['output_folder'])
        if os.path.isdir(folder_to_save):
            shutil.rmtree(folder_to_save)
        if os.path.isdir(os.path.join(path, input_folder)):
            for file in os.listdir(os.path.join(path, input_folder)):
                if '.json' in file:
                    exercise = file.split('mp')[1].split('_')[0]
                    if exercise in exercises:
                        hand = file.split('_')[1].split('.json')[0]
                        values, frame = self.signal_MP(os.path.join(path, input_folder, file), exercise, hand)
                        if len(values) > self.config['automarking']['MP']['threshold_data_length']:
                            maxP, minP, maxA, minA, frac, order_min, order_max = self.auto_point_hand(values, frame)
                            #maxP, minP, maxA, minA = self.signalPoint(maxP, minP, maxA, minA)
                            if self.config['automarking']['image_save']:
                                if not os.path.isdir(os.path.join(output_dir, 'MP')):
                                    os.mkdir(os.path.join(output_dir, 'MP'))
                                path_to_save_image = os.path.join(output_dir, input_folder, file.split('.json')[0]+'png')
                                self.plot_image(values, frame, maxP, minP, maxA, minA, path_to_save_image, file.split('.json')[0])
                            self.write_point_hand(folder_to_save, file, maxP, minP, maxA, minA, frac, order_min, order_max)

    def auto_point_MP(self, values, frame):
        auto_alg_class = instantiate(self.config['automarking']['MP']['auto_alg_class'])
        maxP, minP, maxA, minA, frac, order_min, order_max = auto_alg_class.get_point(values, frame)
        return maxP, minP, maxA, minA, frac, order_min, order_max

    def signal_exersice_MP(self, data, hand, exersice):
        if exersice=='1':
            values, frame = self.signal_FT_angle(data, hand)
        if exersice=='2':
            values, frame = self.signal_OC_angle(data, hand)
        if exersice=='3':
            values, frame = self.signal_PS_mp(data, hand)
        return values, frame
    

    def compute_angle(self, data, point1, point2, vertex_point, y_norm=1,  ):

        x1 = data[point1]['X1'] - data[vertex_point]['X']
        y1 = data[point1]['Y1'] - data[vertex_point]['Y']
        z1 = data[point1]['Z1'] - data[vertex_point]['Z']

        x2 = data[point2]['X1'] - data[vertex_point]['X']
        y2 = data[point2]['Y1'] - data[vertex_point]['Y']
        z2 = data[point2]['Z1'] - data[vertex_point]['Z']

        len1 = np.sqrt(x1**2 + y_norm*y1**2 + z1**2)
        len2 = np.sqrt(x2**2 + y_norm*y2**2 + z2**2)

        cos_val = (x1*x2 + y1*y2 + z1*z2) / (len1 * len2)
        
        return np.arccos(cos_val)*180/np.pi
    

    #получение сигналов двигательной активности рук


    def signal_MP(self, file, exersice, hand_type):
        hand_dict = {'L':'left hand', 'R':'right hand'}
        hand = hand_dict[hand_type]
        print(file)
        data = json.load(open(file))
        values, frame = self.signal_exersice_MP(data, hand, exersice)
        if len(values) < self.config['automarking']['MP']['threshold_data_length']: #TODO #if len(values)==0:
            del hand_dict[hand_type]
            hand = hand_dict[list(hand_dict.keys())[0]]
            values, frame = self.signal_exersice_MP(data, hand, exersice)
        return values, frame
    
    def signal_FT_angle(self, data, hand):
        frame = []
        values = []
        point1 = 'THUMB_TIP'
        point2 = 'FORE_TIP'
        vertex_point = 'THUMB_MCP'
        for i in range(len(data)):
            if hand in data[i].keys():
                distance = self.compute_angle(data[i][hand], point1, point2, vertex_point)
                values.append(distance)
                frame.append(data[i][self.config['automarking']['MP']['timestamp']])
        return values, frame
    

    def signal_OC_angle(self, data, hand):
        frame = []
        values = []
        point1 = 'CENTRE'
        point2 = 'MIDDLE_TIP'
        vertex_point = 'MIDDLE_MCP'
        for i in range(len(data)):
            if hand in data[i].keys():
                distance =  self.compute_angle(data[i][hand], point1, point2, vertex_point)
                values.append(distance)
                frame.append(data[i][self.config['automarking']['MP']['timestamp']])
        return values, frame
    

    def signal_PS_mp(self, data, hand):
    # x1 = points[5]['x'] - points[0]['x']
    # y1 = points[5]['y'] - points[0]['y']
    # z1 = points[5]['z'] - points[0]['z']
        frame = []
        values = []
        point1 = 'LITTLE_TIP'
        point2 = 'RING_TIP'
        for i in range(len(data)):
            if hand in data[i].keys():
                x1 = data[i][hand][point1]['X1'] - data[i][hand][point2]['X1']
                y1 = data[i][hand][point1]['Y1'] - data[i][hand][point2]['Y1']
                z1 = data[i][hand][point1]['Z1'] - data[i][hand][point2]['Z1']

                x2 = 0
                y2 = 1
                z2 = 0

                len1 = np.sqrt(x1**2 + y1**2 + z1**2)
                len2 = np.sqrt(x2**2 + y2**2 + z2**2)

                cos_val = (x1*x2 + y1*y2 + z1*z2) / (len1 * len2)
                distance = np.arccos(cos_val)*180/np.pi
                values.append(distance)
                frame.append(data[i][self.config['automarking']['MP']['timestamp']])
        return values, frame
    

    def processing(self, output_dir):
        df = self.data_processing()
        for path in df['path']:
            for mode in self.config['automarking']['mode']:
                if mode=='hand':
                    self.leap_processing_auto_point(path, output_dir)
                if mode=='face':
                    self.face_processing_auto_point(path, output_dir)
                if mode=='MP':
                    self.MP_processing_auto_point(path, output_dir)