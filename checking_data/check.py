import pandas as pd
import os
import re
import numpy as np
import json
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import cv2
from data_base.hand3D import HandData
from data_base.hand2D import HandDataAngle
from hydra.utils import instantiate
import math

class CheckData():
    def __init__(self, config):
        self.config = config

    def _check_file_exist(self, path, file):
        if os.path.isfile(os.path.join(path, file)):
            file_size = os.stat(os.path.join(path, file)).st_size
            file_flag = True
        else:
            file_size = np.NaN
            file_flag = False
        return file_size, file_flag

    def _check_json_len(self, path, m, file):
        files_json = os.listdir(os.path.join(path, self.config[self.mode]['folder']))
        file_json = [file_j for file_j in files_json if file.split(self.config[self.mode]['dtype'])[0] + '_' + m in file_j]
        if len(file_json) != 0:
            file_json_name = file_json[0]
            file_size_json = os.stat(os.path.join(path, self.config[self.mode]['folder'], file_json_name)).st_size
            data = json.load(open(os.path.join(path, self.config[self.mode]['folder'], file_json_name)))
            json_length = len(data)
            hand_L_in_json = 0
            hand_R_in_json = 0
            for i in range(len(data)):
                if 'left hand' in data[i]:
                    hand_L_in_json = hand_L_in_json + 1
                if 'right hand' in data[i]:
                    hand_R_in_json = hand_R_in_json + 1
            return file_json_name, file_size_json, json_length, hand_L_in_json, hand_R_in_json
        else:
            return np.NaN,np.NaN,np.NaN,np.NaN,np.NaN

    def _check_point(self, path_to_dir, folder, r, file, m, folder_point):
        path = os.path.join(path_to_dir, folder, r)
        if os.path.isdir(os.path.join(path, folder_point)):
            files_point = os.listdir(os.path.join(path, folder_point))
            file_point = [file_m for file_m in files_point if file.split(self.config[self.mode]['dtype'])[0] + '_' + m in file_m]
            if len(file_point) != 0:
                file_point_name = file_point[0]
                path_to_point = os.path.join(path_to_dir, folder, r, folder_point, file_point_name)
                data = json.load(open(path_to_point))
                point_length = len(data)
                signal_x = []
                for i in range(len(data)):
                    signal_x.append(data[i]['X'])
                point_X = max(signal_x)
            else:
                file_point_name, point_length, point_X = np.NaN,np.NaN,np.NaN
        else:
            file_point_name, point_length, point_X = np.NaN,np.NaN,np.NaN

        return file_point_name, point_length, point_X


    def plot_hand_signals_and_points(self, output_dir, path_to_dir, file_signal, file_mannual_point, file_auto_point):
        point_mannual = []
        point_auto = []
        print(path_to_dir,file_signal)
        exersice = file_signal.split('_')[0].split(self.config[self.mode]['prefix'])[1]
        hand_type = file_signal.split('_')[1]
        hand_class = instantiate(self.config[self.mode]['hand_class'])
        values, frame, _ = hand_class.signal_hand(os.path.join(path_to_dir, self.config[self.mode]['folder'], file_signal), exersice, hand_type)
        if os.path.isfile(os.path.join(path_to_dir, self.config[self.mode]['path_to_mannual_point'], str(file_mannual_point))):
            point_mannual = json.load(open(os.path.join(path_to_dir, self.config[self.mode]['path_to_mannual_point'], str(file_mannual_point))))
        if os.path.isfile(os.path.join(path_to_dir, self.config[self.mode]['path_to_auto_point'], str(file_auto_point))):
            point_auto = json.load(open(os.path.join(path_to_dir, self.config[self.mode]['path_to_auto_point'], str(file_auto_point))))
        figure(figsize=(20, 6), dpi=80)
        fig, axs = plt.subplots(2, figsize=(20, 12))
        axs[0].plot(frame, values)
        axs[0].set_title('Mannual ' + str(file_mannual_point), fontsize=15)
        axs[0].set_xlabel('Time (sec)', fontsize=15)
        axs[0].set_ylabel('Distance (mm)', fontsize=15)
        for i in range(len(point_mannual)):
            if point_mannual[i]['Type'] == 0:
                axs[0].plot(point_mannual[i]['X'], point_mannual[i]['Y'], 'o', color='blue')
            else:
                axs[0].plot(point_mannual[i]['X'], point_mannual[i]['Y'], 'o', color='red')
        axs[0].tick_params(axis='x', labelsize=15)
        axs[0].tick_params(axis='y', labelsize=15)
        axs[1].plot(frame, values)
        axs[1].set_title('Auto ' + str(file_auto_point), fontsize=15)
        axs[1].set_xlabel('Time (sec)', fontsize=15)
        axs[1].set_ylabel('Distance (mm)', fontsize=15)
        for i in range(len(point_auto)):
            if point_auto[i]['Type'] == 0:
                axs[1].plot(point_auto[i]['X'], point_auto[i]['Y'], 'o', color='blue')
            else:
                axs[1].plot(point_auto[i]['X'], point_auto[i]['Y'], 'o', color='red')
        axs[1].tick_params(axis='x', labelsize=15)
        axs[1].tick_params(axis='y', labelsize=15)
        r = path_to_dir[-1]
        plt.savefig(os.path.join(output_dir, file_signal.split('.')[0] + '_' + r + '.png'))

    def plot_face_signals_and_points(self, output_dir, path_to_dir, file_signal, file_mannual_point, file_auto_point):
        pass #TODO

    def plot_tremor_signals_and_points(self, output_dir, path_to_dir, file_signal):
        pass #TODO
    def hand_verification(self, path_to_dir, id_name, numbers, output_dir):
        result = []
        for folder in [id_name + str(number) for number in numbers]:
            for r in os.listdir(os.path.join(path_to_dir, folder)):
                path = os.path.join(path_to_dir, folder, r)
                folders = os.listdir(path)
                ms = [m for m in folders if re.findall(r'm\d+', m)]
                for m in ms:
                    if os.path.isdir(os.path.join(path, m)):
                        for file in self.config[self.mode]['exercise']:
                            file_size, file_flag = self._check_file_exist(os.path.join(path, m), file)
                            file_json_name, file_size_json, json_length, hand_L_in_json, hand_R_in_json = self._check_json_len(path, m, file)
                            file_mannual_point_name, mannual_point_length, mannual_point_X = self._check_point(path_to_dir, folder, r, file, m, self.config[self.mode]['path_to_mannual_point'])
                            file_auto_point_name, auto_point_length, auto_point_X = self._check_point(path_to_dir, folder, r, file, m, self.config[self.mode]['path_to_auto_point'])
                            result.append({
                                'folder': folder,
                                'r': r,
                                'm': m,
                                'exersise': file.split('_')[0].split(self.config[self.mode]['prefix'])[1],
                                'hand': file.split('_')[1].split(self.config[self.mode]['dtype'])[0],
                                'lmt_type': file,
                                'lmt': file_flag,
                                'lmt_weight': file_size,
                                'json': file_json_name,
                                'mannual_point': file_mannual_point_name,
                                'mannual_point_length': mannual_point_length,
                                'mannual_point_X': mannual_point_X,
                                'auto_point': file_auto_point_name,
                                'auto_point_length': auto_point_length,
                                'auto_point_X': auto_point_X,
                                'json_weight': file_size_json,
                                'json_length': json_length,
                                'hand_L_in_json': hand_L_in_json,
                                'hand_R_in_json': hand_R_in_json,
                            })
                            if (isinstance(file_json_name, str) & (self.config['save_plot'])):
                                if not os.path.isdir(os.path.join(output_dir, 'hand_signal')):
                                    os.mkdir(os.path.join(output_dir, 'hand_signal'))
                                self.plot_hand_signals_and_points(os.path.join(output_dir, 'hand_signal'), path, file_json_name, file_mannual_point_name, file_auto_point_name)

        return pd.DataFrame(result)
    def tremor_verification(self, path_to_dir, id_name, numbers):
        result = []
        for folder in [id_name + str(number) for number in numbers]:
            for r in os.listdir(os.path.join(path_to_dir, folder)):
                path = os.path.join(path_to_dir, folder, r)
                folders = os.listdir(path)
                ms = [m for m in folders if re.findall(r'm\d+', m)]
                for m in ms:
                    if os.path.isdir(os.path.join(path, m)):
                        for file in self.config['tremor']['exercise']:
                            file_size, file_flag = self._check_file_exist(path, m, file)
                            file_json_name, file_size_json, json_length, hand_L_in_json, hand_R_in_json = self._check_json_len(path, m, file)
                            result.append({
                                'folder': folder,
                                'r': r,
                                'm': m,
                                'exersise': file.split('_')[0].split('leapRecording')[1],
                                'hand': file.split('_')[1].split('.lmt')[0],
                                'lmt_type': file,
                                'lmt': file_flag,
                                'lmt_weight': file_size,
                                'json': file_json_name,
                                'json_weight': file_size_json,
                                'json_length': json_length,
                                'hand_L_in_json': hand_L_in_json,
                                'hand_R_in_json': hand_R_in_json,
                            })
            return pd.DataFrame(result)


    def _check_face_file(self, path, folder, ex):
        if os.path.isfile(os.path.join(path, ex + '_' + folder +'.mp4')):
            video_name = os.path.join(path, ex + '_' + folder + '.mp4')
            cap = cv2.VideoCapture(video_name)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            return video_name, frame_count, video_fps, width, height
        else:
            return np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
    def _check_point_face_file(self, path, folder, ex):
        if os.path.isfile(os.path.join(path, ex + '_' + folder +'.csv')):
            file_name = ex + '_' + folder + '.csv'
        else:
            file_name = np.NaN
        if os.path.isfile(os.path.join(path, self.config['face']['path_to_mannual_point'], ex + '_' + folder +'.json')):
            file_mannual_point_name =  ex + '_' + folder +'.json'
        else:
            file_mannual_point_name = np.NaN
        if os.path.isfile(os.path.join(path, self.config['face']['path_to_auto_point'], ex + '_' + folder +'.json')):
            file_auto_point_name = ex + '_' + folder +'.json'
        else:
            file_auto_point_name = np.NaN
        return file_name, file_mannual_point_name, file_auto_point_name

    def face_verification(self, dataset, path_to_dir, id_name, numbers, output_dir):
        result = []
        for folder in [id_name + str(number) for number in numbers]:
            for r in os.listdir(os.path.join(path_to_dir, folder)):
                path = os.path.join(path_to_dir, folder, r, 'face')
                for ex in self.config['face']['exercise'].keys():
                    video_name, frame_count, video_fps, width, height  = self._check_face_file(path, folder, ex)
                    file_name, file_mannual_point_name, file_auto_point_name = self._check_point_face_file(os.path.join(path_to_dir, folder, r), folder, ex)

                    result.append({
                        'folder': folder,
                        'r': r,
                        'exersise': ex,
                        'file_name':file_name,
                        'video_name':video_name,
                        'frame_count':frame_count,
                        'video_fps':video_fps,
                        'width':width,
                        'height':height,
                        'file_mannual_point_name':file_mannual_point_name,
                        'file_auto_point_name':file_mannual_point_name
                    })

        return pd.DataFrame(result)

    def em_verification(self, dataset): #TODO
        pass

    def processing(self, output_dir):
        for mode in self.config['mode']:
            self.mode = mode
            df_result = []
            df = pd.DataFrame()
            for dataset in ['PD','HEALTHY','STUDENT']:
                path_to_dir = self.config[dataset]['path_to_directory']
                id_name = self.config[dataset]['id_name']
                numbers = self.config[dataset]['number']
                if mode=='hand':
                    df = self.hand_verification(path_to_dir,id_name,numbers,output_dir)
                if mode=='hand2D':
                    df = self.hand_verification(path_to_dir,id_name,numbers,output_dir)
                if mode=='tremor':
                    df = self.tremor_verification(path_to_dir,id_name,numbers)
                if mode == 'face':
                    df = self.face_verification(dataset,path_to_dir,id_name,numbers,os.path.join(output_dir, 'face_signal'))
                df['dataset'] = dataset
                df_result.append(df)

                '''
                if mode=='em':
                    df_face.append(self.em_verification(dataset))
                '''
            pd.concat(df_result).to_csv(os.path.join(output_dir, mode + '_verification.csv'))
