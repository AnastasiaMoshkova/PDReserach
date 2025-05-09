import pandas as pd
import os
import re
import numpy as np
import json
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import cv2
from data_base.hand3D import HandData
from data_base.tremor import TremorProcessing
import math

class SignalVisualization(HandData):
    def __init__(self, config):
        self.config = config

    def plot_hand_signals(self, output_dir, task, path_to_file):
        print(path_to_file)
        file_json = [f for f in os.listdir(path_to_file) if ((f.split('_')[0][-1]==task)&(f.endswith('.json')))]
        for file in file_json:
            exersice = file.split('_')[0].split('leapRecording')[1]
            hand_type = file.split('_')[1].split('.')[0]
            values, frame, palm_width = self.signal_hand(os.path.join(path_to_file, file), exersice, hand_type)
            if ((task=='1')|(task=='2')):
                values = values/np.mean(palm_width)
            df = pd.DataFrame({'values':values, 'frame':frame})
            df = df[((df['frame']<self.config['hand']['length_threshold'][1])&(df['frame']>self.config['hand']['length_threshold'][0]))]
            figure(figsize=(20, 6), dpi=80)
            plt.plot(df['frame'], df['values'])
            #plt.title(file, fontsize=15)
            plt.xlabel('Time (sec)', fontsize=15)
            plt.ylabel('Distance (mm)', fontsize=15)
            if ((task == '1') | (task == '2')):
                plt.ylim(0, 1.5)
            if (task == '3'):
                plt.ylim(-70, 150)
            plt.savefig(os.path.join(output_dir, file.split('.')[0]+'.png'))
    def hand_signal_plot(self, path_to_dir, id_name, numbers, output_dir):
        for folder in [id_name + str(number) for number in numbers]:
            for r in os.listdir(os.path.join(path_to_dir, folder)):
                path = os.path.join(path_to_dir, folder, r)
                if os.path.isdir(os.path.join(path, self.config['hand']['folder'])):
                    for task in self.config['hand']['exercise']:
                        if not os.path.isdir(os.path.join(output_dir, 'hand_signal')):
                            os.mkdir(os.path.join(output_dir, 'hand_signal'))
                        self.plot_hand_signals(os.path.join(output_dir, 'hand_signal'),task, os.path.join(path, self.config['hand']['folder']))

    #TODO refactor
    def tremor_signal_plot(self, path_to_dir, id_name, numbers, output_dir):
        for folder in [id_name + str(number) for number in numbers]:
            for r in os.listdir(os.path.join(path_to_dir, folder)):
                path = os.path.join(path_to_dir, folder, r)
                if os.path.isdir(os.path.join(path, self.config['tremor']['folder'])):
                    if not os.path.isdir(os.path.join(output_dir, 'tremor_signal')):
                        os.mkdir(os.path.join(output_dir, 'tremor_signal'))
                    output=os.path.join(output_dir, 'tremor_signal')
                    file_json = [f for f in os.listdir(os.path.join(path, self.config['tremor']['folder'])) if
                                 (('TR' in f) & (f.endswith('.json')))]
                    for file in file_json:
                        if 'RL_' in file:
                            TremorProcessing().plot_tremor(os.path.join(path, self.config['tremor']['folder']), file, 'R', self.config['tremor']['key_point'], output)
                            TremorProcessing().plot_tremor(os.path.join(path, self.config['tremor']['folder']), file, 'L', self.config['tremor']['key_point'], output)
                        if '_R_' in file:
                            TremorProcessing().plot_tremor(os.path.join(path, self.config['tremor']['folder']), file, 'R', self.config['tremor']['key_point'], output)
                        if '_L_' in file:
                            TremorProcessing().plot_tremor(os.path.join(path, self.config['tremor']['folder']), file, 'L',self.config['tremor']['key_point'], output)

    def processing(self, output_dir):
        for mode in self.config['mode']:
            for dataset in ['PD','HEALTHY','STUDENT']:
                path_to_dir = self.config[dataset]['path_to_directory']
                id_name = self.config[dataset]['id_name']
                numbers = self.config[dataset]['number']
                if mode=='hand':
                    self.hand_signal_plot(path_to_dir, id_name, numbers, output_dir)
                if mode=='tremor':
                    self.tremor_signal_plot(path_to_dir, id_name, numbers, output_dir)

