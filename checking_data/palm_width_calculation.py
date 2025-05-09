import os
import numpy as np
import json

class PalmWidth:
    def __init__(self, config):
        self.config = config

    def _palm_width_calculation(self, path, hand):
        with open(path) as f:
            data = json.load(f)
        palm_width = []
        for i in range(len(data)):
            if hand in data[i].keys():
                palm_width.append(data[i][hand]['info']['palm_width'])
        return palm_width

    def processing(self):
        for dataset in ['PD', 'HEALTHY', 'STUDENT']:
            path_init = self.config[dataset]['path_to_directory']
            id_name = self.config[dataset]['id_name']
            users = self.config[dataset]['number']
            for folder in os.listdir(path_init):
                if int(folder.split(id_name)[1]) in users:
                    for r in os.listdir(os.path.join(path_init, folder)):
                        if self.config['hand']['folder_signals'] in os.listdir(os.path.join(path_init, folder, r)):
                            print(os.path.join(path_init, folder, r, self.config['hand']['folder_signals']))
                            hand_coeff = []
                            for file in os.listdir(os.path.join(path_init, folder, r, self.config['hand']['folder_signals'])):
                                if (('.json' in file) & ('leapRecording2_L' in file)):
                                    path = os.path.join(path_init, folder, r, self.config['hand']['folder_signals'], file)
                                    hand_coeff.extend(self._palm_width_calculation(path, 'left hand'))

                                if (('.json' in file) & ('leapRecording2_R' in file)):
                                    path = os.path.join(path_init, folder, r, self.config['hand']['folder_signals'], file)
                                    hand_coeff.extend(self._palm_width_calculation(path, 'right hand'))

                            if len(hand_coeff) > 0:
                                palm_width = {
                                    'palm_width_mean': np.mean(hand_coeff),
                                    'palm_width_max': np.max(hand_coeff),
                                    'palm_width_std': np.std(hand_coeff), }


                                if not self.config['hand']['folder_to_save'] in os.listdir(os.path.join(path_init, folder, r)):
                                    os.mkdir(os.path.join(path_init, folder, r, self.config['hand']['folder_to_save']))

                                out_file = open(
                                    os.path.join(path_init, folder, r, self.config['hand']['folder_to_save'], 'palm_width.json'), "w")
                                json.dump(palm_width, out_file)

