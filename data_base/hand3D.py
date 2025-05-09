import os
import pandas as pd
import json
import math
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from data_base.hand import HandBase

TIMESTAMP_COEFFICIENT = 1000000
class HandData(HandBase):


    def signal_exersice_hand(self, data, hand, exersice):
        if exersice=='1':
            values, frame, palm_width = self.signal_FT(data, hand)
        if exersice=='2':
            values, frame, palm_width = self.signal_OC(data, hand)
        if exersice=='3':
            values, frame, palm_width = self.signal_PS(data, hand)
        return values, frame, palm_width


    def signal_FT(self, data, hand):
        frame = []
        values = []
        timestamps = []
        palm_width = []
        for i in range(len(data)):
            if hand in data[i].keys():
                sum_sqr = ((float(data[i][hand]['FORE_TIP']['X1']) - float(data[i][hand]['THUMB_TIP']['X1'])) ** 2 +
                           (float(data[i][hand]['FORE_TIP']['Y1']) - float(data[i][hand]['THUMB_TIP']['Y1'])) ** 2 +
                           (float(data[i][hand]['FORE_TIP']['Z1']) - float(data[i][hand]['THUMB_TIP']['Z1'])) ** 2)
                distance = math.sqrt(sum_sqr)
                values.append(distance)
                frame.append(data[i]['frame'])
                timestamps.append((data[i][hand]['info']['timestamp']) / TIMESTAMP_COEFFICIENT)
                palm_width.append(data[i][hand]['info']['palm_width'])
        timestamps = [ts - timestamps[0] for ts in timestamps]
        return values, timestamps, palm_width

    def signal_OC(self, data, hand):
        frame = []
        values = []
        timestamps = []
        palm_width = []
        for i in range(len(data)):
            if hand in data[i].keys():
                sum_sqr = (float(data[i][hand]['MIDDLE_TIP']['X1']) - float(data[i][hand]['CENTRE']['X'])) ** 2 + (
                                  float(data[i][hand]['MIDDLE_TIP']['Y1']) - float(data[i][hand]['CENTRE']['Y'])) ** 2 + (
                                      float(data[i][hand]['MIDDLE_TIP']['Z1']) - float(data[i][hand]['CENTRE']['Z'])) ** 2
                distance = math.sqrt(sum_sqr)
                values.append(distance)
                frame.append(data[i]['frame'])
                timestamps.append((data[i][hand]['info']['timestamp']) / TIMESTAMP_COEFFICIENT)
                palm_width.append(data[i][hand]['info']['palm_width'])
        timestamps = [ts - timestamps[0] for ts in timestamps]
        return values, timestamps, palm_width

    def signal_PS(self, data, hand):
        frame = []
        values = []
        timestamps = []
        for i in range(len(data)):
            if hand in data[i].keys():
                values.append(float(data[i][hand]['CENTRE']['Angle']))
                frame.append(data[i]['frame'])
                timestamps.append((data[i][hand]['info']['timestamp']) / TIMESTAMP_COEFFICIENT)
        timestamps = [ts - timestamps[0] for ts in timestamps]
        return values, timestamps, []





