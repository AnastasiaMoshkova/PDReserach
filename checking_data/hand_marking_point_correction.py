import os
import json
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np


#TODO code refactoring
class PointCorrection:
    def __init__(self, config):
        self.config = config

    def _plot_hand(self, path, task, hand):
        with open(path) as f:
            data = json.load(f)
        timestamps = []
        distance = []
        confidence = []
        id_frame = []
        visible_time = []
        pinch_distance = []
        palm_width = []
        frame_id = []
        tracking_frame_id = []
        framerate = []
        frame = []
        for i in range(len(data)):
            if hand in data[i].keys():
                timestamps.append((data[i][hand]['info']['timestamp']) / 1000000)
                if task == 1:
                    sum_sqr = ((float(data[i][hand]['FORE_TIP']['X1']) - float(data[i][hand]['THUMB_TIP']['X1'])) ** 2 +
                               (float(data[i][hand]['FORE_TIP']['Y1']) - float(data[i][hand]['THUMB_TIP']['Y1'])) ** 2 +
                               (float(data[i][hand]['FORE_TIP']['Z1']) - float(data[i][hand]['THUMB_TIP']['Z1'])) ** 2)
                    distance.append(math.sqrt(sum_sqr))
                if task == 2:
                    sum_sqr = (float(data[i][hand]['MIDDLE_TIP']['X1']) - float(data[i][hand]['CENTRE']['X'])) ** 2 + (
                            float(data[i][hand]['MIDDLE_TIP']['Y1']) - float(data[i][hand]['CENTRE']['Y'])) ** 2 + (
                                      float(data[i][hand]['MIDDLE_TIP']['Z1']) - float(
                                  data[i][hand]['CENTRE']['Z'])) ** 2
                    distance.append(math.sqrt(sum_sqr))
                if task == 3:
                    distance.append(float(data[i][hand]['CENTRE']['Angle']))

                confidence.append(data[i][hand]['info']['confidence'] * 100)
                id_frame.append(data[i][hand]['info']['id_frame'])
                visible_time.append((data[i][hand]['info']['visible_time']) / 1000000)
                pinch_distance.append(data[i][hand]['info']['pinch_distance'])
                palm_width.append(data[i][hand]['info']['palm_width'])
                frame_id.append(data[i][hand]['info']['frame_id'] / 1000000000000)
                tracking_frame_id.append(data[i][hand]['info']['tracking_frame_id'])
                framerate.append(data[i][hand]['info']['framerate'])
                frame.append(data[i]['frame'])
        # print(timestamps[0],timestamps[1], timestamps[1] - timestamps[0])
        timestamps = list(map(lambda x: x - timestamps[0], timestamps))
        fps = np.mean(framerate)
        frame2 = list(map(lambda x: x / fps, frame))
        frame = list(map(lambda x: x / 100, frame))
        if len(framerate) > 1:
            return distance, timestamps, frame, [min(framerate), np.mean(framerate),
                                                 max(framerate)], confidence, frame2, fps
        else:
            return distance, timestamps, frame, [framerate], confidence, frame2, fps
    def processing(self, path_to_save):
        for dataset in ['PD', 'HEALTHY', 'STUDENT']:
            path_init = self.config[dataset]['path_to_directory']
            id_name = self.config[dataset]['id_name']
            users = self.config[dataset]['number']
            for folder in os.listdir(path_init):
                if int(folder.split(id_name)[1]) in users:
                    for r in os.listdir(os.path.join(path_init, folder)):
                        if self.config['hand']['folder_signals'] in os.listdir(os.path.join(path_init, folder, r)):
                            for file in os.listdir(os.path.join(path_init, folder, r, self.config['hand']['folder_signals'] )):
                                if (('.json' in file) & ('TR' not in file) & ('3G' not in file)):
                                    path = os.path.join(path_init, folder, r, self.config['hand']['folder_signals'], file)
                                    path_point = os.path.join(path_init, folder, r, self.config['hand']['folder_in'], file)
                                    maxPointX = []
                                    minPointX = []
                                    maxPointY = []
                                    minPointY = []
                                    for p in os.listdir(os.path.join(path_init, folder, r, self.config['hand']['folder_in'])):
                                        if file.split('.')[0] in p:
                                            file_point = p
                                            datapoint = json.load(open(os.path.join(path_init, folder, r, self.config['hand']['folder_in'], p)))
                                            datapoint = sorted(datapoint, key=lambda k: k['X'])
                                            maxPointX = []
                                            minPointX = []
                                            maxPointY = []
                                            minPointY = []
                                            for point in datapoint:
                                                if point['Type'] == 1:
                                                    maxPointX.append(point['X'] / 100)
                                                    maxPointY.append(point['Y'])
                                                if point['Type'] == 0:
                                                    # if ((point['Type']==0)&(point['X']>940))|((point['Type']==0)&(910>point['X'])):
                                                    minPointX.append(point['X'] / 100)
                                                    minPointY.append(point['Y'])

                                    if '_L_' in file:
                                        hand = 'left hand'
                                    if '_R_' in file:
                                        hand = 'right hand'
                                    task = int(file.split('leapRecording')[1].split('_')[0])
                                    distance, timestamps, frame, fps, confidence, frame2, fps = self._plot_hand(path, task, hand)
                                    if len(frame) < 500:
                                        if '_L_' in file:
                                            hand = 'right hand'
                                        if '_R_' in file:
                                            hand = 'left hand'
                                        print('HAND----------------------------------------')
                                        distance, timestamps, frame, fps, confidence, frame2, fps = self._plot_hand(path, task, hand)

                                    if frame[0] > 1:
                                        frame = [frame[i] - frame[0] for i in range(len(frame))]
                                        print('FRAME0----------------------------------------')
                                    d = dict(zip(frame, timestamps))
                                    print(folder, r, 'handv22', path, fps)
                                    df = pd.DataFrame(d, index=['timestamp']).transpose()
                                    df['delta'] = df['timestamp'].diff()
                                    if len(df[df['delta'] > 1]) > 0:
                                        df['timestamp'] = [frame[i] * 100 / fps for i in range(len(frame))]
                                        print('DELTA----------------------------------------')
                                    # df.loc[df['delta']>1, 'timestamp'] = np.nan
                                    df.loc[df['timestamp'] < 0, 'timestamp'] = np.nan
                                    if np.isnan(df.iloc[-1]['timestamp']):
                                        df['timestamp'] = [frame[i] * 100 / fps for i in range(len(frame))]
                                        print('NAN----------------------------------------')
                                    # df = df.interpolate(method="spline", limit_direction='forward', axis=0)
                                    # print(list(df['timestamp'].values))
                                    figure(figsize=(32, 8), dpi=80)
                                    ax1 = plt.subplot(3, 1, 1)
                                    ax1.plot(frame, distance, label='frame')
                                    ax1.plot(maxPointX, maxPointY, 'ro')
                                    ax1.plot(minPointX, minPointY, 'bo')
                                    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
                                    ax2.plot(df['timestamp'], distance, label='timestamp')
                                    # print(df.index.get_indexer(12, method='nearest'))
                                    try:
                                        if (np.isnan(df.iloc[-1]['timestamp']) | len(df[df['delta'] > 1]) > 0):
                                            ax2.plot([maxp * 100 / fps for maxp in maxPointX], maxPointY, 'ro')
                                            ax2.plot([minp * 100 / fps for minp in minPointX], minPointY, 'bo')
                                            maxPointX_new = [maxp * 100 / fps for maxp in maxPointX]
                                            minPointX_new = [minp * 100 / fps for minp in minPointX]
                                        else:
                                            ax2.plot(
                                                [df.iloc[df.index.get_indexer([maxp], method='nearest')]['timestamp'].values[0] for
                                                 maxp in maxPointX], maxPointY, 'ro')
                                            ax2.plot(
                                                [df.iloc[df.index.get_indexer([minp], method='nearest')]['timestamp'].values[0] for
                                                 minp in minPointX], minPointY, 'bo')
                                            maxPointX_new = [
                                                df.iloc[df.index.get_indexer([maxp], method='nearest')]['timestamp'].values[0] for maxp in maxPointX]
                                            minPointX_new = [
                                                df.iloc[df.index.get_indexer([minp], method='nearest')]['timestamp'].values[0] for minp in minPointX]
                                        # ax2.plot([df.loc[maxp] for maxp in maxPointX], maxPointY, 'ro')
                                        # ax2.plot([df.loc[minp] for minp in minPointX], minPointY, 'bo')
                                        dpoint = []
                                        if len(maxPointX_new) != 0:
                                            for i in range(len(maxPointX_new)):
                                                dpoint.append({'Type': 1,
                                                               'Scale': 1.0,
                                                               'Brush': '#FFFF0000',
                                                               'X': maxPointX_new[i],
                                                               'Y': maxPointY[i]})
                                        if len(minPointX_new) != 0:
                                            for i in range(len(minPointX_new)):
                                                dpoint.append({'Type': 0,
                                                               'Scale': 1.0,
                                                               'Brush': '#FF0000FF',
                                                               'X': minPointX_new[i],
                                                               'Y': minPointY[i]})
                                            if not os.path.exists(os.path.join(path_init, folder, r, self.config['hand']['folder_out'])):
                                                os.mkdir(os.path.join(path_init, folder, r, self.config['hand']['folder_out']))
                                            with open(os.path.join(os.path.join(path_init, folder, r, self.config['hand']['folder_out'], file_point)), 'w') as f:
                                                json.dump(dpoint, f)
                                    except Exception as e:
                                        print('error', e)
                                    '''
                                    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
                                    ax3.plot(timestamps, distance, label = 'timestamp')
                                    try:
                                        ax3.plot([d[maxp] for maxp in maxPointX], maxPointY, 'ro')
                                        ax3.plot([d[minp] for minp in minPointX], minPointY, 'bo')
                                    except Exception as e:
                                        print(e)
                                    '''
                                    # plt.plot(frame, pinch_distance, label = 'pinch')
                                    # plt.plot(timestamps, confidence, label = 'confidence')
                                    # plt.plot(frame2, distance, label = 'frame2')
                                    plt.xlabel('frame')
                                    plt.ylabel('Distance')
                                    plt.legend()
                                    plt.savefig(os.path.join(path_to_save, file_point.split('.')[0]+'.png'))
                                    #plt.show()