import os
import pandas as pd
import json
import math
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

class HandData:

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

    def signal_exersice_hand(self, data, hand, exersice):
        if exersice=='1':
            values, frame = self.signal_FT(data, hand)
        if exersice=='2':
            values, frame = self.signal_OC(data, hand)
        if exersice=='3':
            values, frame = self.signal_PS(data, hand)
        return values, frame

    def signal_hand(self, file, exersice, hand_type):
        hand_dict = {'L':'left hand', 'R':'right hand'}
        hand = hand_dict[hand_type]
        data = json.load(open(file))
        values, frame = self.signal_exersice_hand(data, hand, exersice)
        if len(values)==0:
            del hand_dict[hand_type]
            hand = hand_dict[list(hand_dict.keys())[0]]
            values, frame = self.signal_exersice_hand(data, hand, exersice)
        return values, frame


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
                frame.append(data[i]['frame'])
        return values, frame

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
                frame.append(data[i]['frame'])
        return values, frame

    def signal_PS(self, data, hand):
        frame = []
        values = []
        for i in range(len(data)):
            if hand in data[i].keys():
                values.append(float(data[i][hand]['CENTRE']['Angle']))
                frame.append(data[i]['frame'])
        return values, frame




