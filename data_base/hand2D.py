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

class HandDataAngle(HandBase):

    def compute_angle(self, data, point1, point2, vertex_point, y_norm=1, ):

        x1 = data[point1['point']][point1['X']] - data[vertex_point['point']][vertex_point['X']]
        y1 = data[point1['point']][point1['Y']] - data[vertex_point['point']][vertex_point['Y']]
        z1 = data[point1['point']][point1['Z']] - data[vertex_point['point']][vertex_point['Z']]

        x2 = data[point2['point']][point2['X']] - data[vertex_point['point']][vertex_point['X']]
        y2 = data[point2['point']][point2['Y']] - data[vertex_point['point']][vertex_point['Y']]
        z2 = data[point2['point']][point2['Z']] - data[vertex_point['point']][vertex_point['Z']]

        len1 = np.sqrt(x1 ** 2 + y_norm * y1 ** 2 + z1 ** 2)
        len2 = np.sqrt(x2 ** 2 + y_norm * y2 ** 2 + z2 ** 2)

        cos_val = (x1 * x2 + y1 * y2 + z1 * z2) / (len1 * len2)

        return np.arccos(cos_val) * 180 / np.pi

    def signal_exersice_hand(self, data, hand, exersice):
        if exersice=='1':
            values, frame, palm_width = self.signal_FT_angle(data, hand)
        if exersice=='2':
            values, frame, palm_width = self.signal_OC_angle(data, hand)
        if exersice=='3':
            values, frame, palm_width = self.signal_PS_angle(data, hand)
        return values, frame, palm_width


    def signal_FT_angle(self, data, hand):
        timestamps = []
        frame = []
        values = []
        point1 = {'point': 'THUMB_TIP',  'X': 'X1', 'Y': 'Y1', 'Z': 'Z1'}
        point2 = {'point': 'FORE_TIP', 'X': 'X1', 'Y': 'Y1', 'Z': 'Z1'}
        vertex_point = {'point':'THUMB_MCP','X': 'X1', 'Y': 'Y1', 'Z': 'Z1'}
        for i in range(len(data)):
            if hand in data[i].keys():
                distance = self.compute_angle(data[i][hand], point1, point2, vertex_point)
                values.append(distance)
                frame.append(data[i]['frame'])
                timestamps.append((data[i][hand]['info']['timestamp']) / TIMESTAMP_COEFFICIENT)
        timestamps = [ts - timestamps[0] for ts in timestamps]
        return values, timestamps, 1


    def signal_OC_angle(self, data, hand):
        timestamps = []
        frame = []
        values = []
        point1 = {'point':'CENTRE', 'X': 'X', 'Y': 'Y', 'Z': 'Z'}
        point2 = {'point':'MIDDLE_TIP','X': 'X1', 'Y': 'Y1', 'Z': 'Z1'}
        vertex_point = {'point':'MIDDLE_MCP','X': 'X1', 'Y': 'Y1', 'Z': 'Z1'}
        for i in range(len(data)):
            if hand in data[i].keys():
                distance = self.compute_angle(data[i][hand], point1, point2, vertex_point)
                values.append(distance)
                frame.append(data[i]['frame'])
                timestamps.append((data[i][hand]['info']['timestamp']) / TIMESTAMP_COEFFICIENT)
        timestamps = [ts - timestamps[0] for ts in timestamps]
        return values, timestamps, 1

    def signal_PS_angle(self, data, hand):
        timestamps = []
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

                len1 = np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)
                len2 = np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)

                cos_val = (x1 * x2 + y1 * y2 + z1 * z2) / (len1 * len2)
                distance = np.arccos(cos_val) * 180 / np.pi
                values.append(distance)
                frame.append(data[i]['frame'])
                timestamps.append((data[i][hand]['info']['timestamp']) / TIMESTAMP_COEFFICIENT)
        timestamps = [ts - timestamps[0] for ts in timestamps]
        return values, timestamps, 1




