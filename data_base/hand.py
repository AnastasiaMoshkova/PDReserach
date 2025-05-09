import os
import pandas as pd
import json
import math
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from abc import ABC, abstractmethod

TIMESTAMP_COEFFICIENT = 1000000

class HandBase(ABC):
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

    def signal_hand(self, file, exersice, hand_type):
        hand_dict = {'L': 'left hand', 'R': 'right hand'}
        hand = hand_dict[hand_type]
        data = json.load(open(file))
        values, frame, palm_width = self.signal_exersice_hand(data, hand, exersice)
        if len(values) == 0:
            del hand_dict[hand_type]
            hand = hand_dict[list(hand_dict.keys())[0]]
            values, frame, palm_width = self.signal_exersice_hand(data, hand, exersice)
        return values, frame, palm_width

    @abstractmethod
    def signal_exersice_hand(self, data, hand, exersice):
        pass