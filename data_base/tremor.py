import json
import numpy as np
from scipy import fftpack
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import loadtxt, optimize
from scipy.signal import argrelmax
from scipy.signal import argrelmin
from scipy.signal import medfilt
from scipy.signal import savgol_filter
from pylab import *
# import pywt
from scipy.fftpack import rfft, irfft, fftfreq
import os
import glob
import numpy
import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
# пострроение спектра сигнала
from scipy import fftpack
from scipy.fftpack import rfft, irfft, fftfreq
from scipy.signal import argrelmax
import scipy as sp
import matplotlib.pyplot as plt


class TremorProcessing:
    '''
    Objective:
        Функция для построения сигналов перемещения ключевой точки руки по X,Y,Z координатам
        при выполнении упражнения TR (запись тремора с датчиком Leap Motion в течении 20 секунд)

    Inputs:
        file: путь к .json файлу с записью
        hand: тип руки - 'left hand' или 'right hand'
        key_point: клюучевая точки ладони, относительно которой сяитаются признаки

    Output:
        X: массив координат по х на каждом кадре
        Y: массив координат по y на каждом кадре
        Z: массив координат по z на каждом кадре
        frame: массив с номерами кадров
    '''

    def _tremor_signal(self, file, hand_type, key_point):
        hands = {'R': 'right hand', 'L': 'left hand'}
        hand = hands[hand_type]
        raw_data = json.load(open(file))
        frame = []
        znach = []
        data = []
        X = []
        Y = []
        Z = []
        for i in range(len(raw_data)):
            if hand in raw_data[i].keys():
                data.append(raw_data[i])
        for i in range(len(data) - 1):
            if hand in data[i].keys():
                X.append(float(data[i][hand][key_point]['X1']))
                Y.append(float(data[i][hand][key_point]['Y1']))
                Z.append(float(data[i][hand][key_point]['Z1']))
                frame.append(data[i]['frame'])
        return X, Y, Z, frame

    # функция принимает frame - массив с номерами адров , znach - какой либо из массивов с координатами X,Y или Z
    # функция возвращает pointY - координату точки по Y (красные точки на графике - огибающая спектра),
    # pointX - координату точки по X (красные точки на графике - огибающая спектра)
    #  maxTemp - массив точек pointY, pointX - для построения огибающей,
    # f_signal - массив со значениями гармоник спектр сигнала, ff - массив со значеними частот, fn - граничный отсчет в массиве частот

    '''
    Objective:

    Inputs:
        frame: массив с номерами кадров
        values: какой либо из массивов с координатами X,Y или Z

    Output:
        pointY: координату точки по Y (красные точки на графике - огибающая спектра)
        pointX: координату точки по X (красные точки на графике - огибающая спектра)
        maxTemp: массив точек pointY, pointX - для построения огибающей
        f_signal: массив со значениями гармоник спектр сигнала
        ff: массив со значеними частот
        fn: граничный отсчет в массиве частот
    '''

    def _spectrXYZ(self, frame, values):
        time = np.array(frame) / 100
        signal = np.array(values)
        W = fftfreq(signal.size, d=time[1] - time[0])
        f_signal = rfft(signal)
        fn = 800
        ff = list(abs(W[1:fn] / 2))
        maxTempnp = argrelmax(abs(f_signal[1:fn - 1]), order=5)
        pointY = []
        pointX = []
        maxTemp = maxTempnp[0].tolist()
        for j in range(len(maxTemp)):
            pointY.append(abs(f_signal[1:fn - 1])[maxTemp[j]])
            pointX.append((ff[1:fn - 1])[maxTemp[j]])
        return pointY, pointX, maxTemp, f_signal, ff, fn

    '''
    Objective:
        Функция осуществляет построение спектра сигнала для 3 координат X,Y,Z на одном графике

    Inputs:
        X, Y, Z: массивов с координатами X,Y или Z
        frame: массив с номерами кадров
        name: название файла

    '''

    def _plot_tremor_signals_and_spectrum(self, frame, X, Y, Z, name, output_path):

        X = np.array(X) - min(X)
        Y = np.array(Y) - min(Y)
        Z = np.array(Z) - min(Z)
        pointY_1, pointX_1, maxTemp_1, f_signal_1, ff_1, fn = self._spectrXYZ(frame, X)
        pointY_2, pointX_2, maxTemp_2, f_signal_2, ff_2, fn = self._spectrXYZ(frame, Y)
        pointY_3, pointX_3, maxTemp_3, f_signal_3, ff_3, fn = self._spectrXYZ(frame, Z)

        SMALL_SIZE = 15
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 30
        linewidth1 = 4
        linewidth2 = 12
        # plt.subplots_adjust(wspace=10, hspace=10)
        plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
        # plt.rc('title', fontsize=BIGGER_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

        fig = plt.figure(figsize=(15, 11))
        ax_1 = fig.add_subplot(3, 1, 1)
        ax_1.plot(np.array(frame) / 100, X)
        ax_1.legend('X')
        ax_1.set_ylabel("Амплитуда (мм)")
        ax_1.set_xlim(-1, 20)

        ax_2 = fig.add_subplot(3, 1, 2)
        ax_2.plot(np.array(frame) / 100, Y)
        ax_2.legend('Y')
        ax_2.set_ylabel("Амплитуда (мм)")
        ax_2.set_xlim(-1, 20)

        ax_3 = fig.add_subplot(3, 1, 3)
        ax_3.plot(np.array(frame) / 100, Z)
        ax_3.legend('Z')
        ax_3.set_xlabel("Время (с)")
        ax_3.set_ylabel("Амплитуда (мм)")
        ax_3.set_xlim(-1, 20)

        # plt.savefig(os.path.join(output_path, name + '_signal.png'))
        plt.show()
        plt.clf()

        fig = plt.figure(figsize=(15, 11))
        ax_1 = fig.add_subplot(3, 1, 1)
        ax_1.plot(ff_1[1:fn - 1], abs(f_signal_1[1:fn - 1]))
        ax_1.legend('X')
        ax_1.plot(pointX_1, pointY_1, color='black')
        for j in range(len(maxTemp_1)):
            ax_1.plot((ff_1[1:fn - 1])[maxTemp_1[j]], abs(f_signal_1[1:fn - 1])[maxTemp_1[j]], 'ro')

        ax_2 = fig.add_subplot(3, 1, 2)
        ax_2.plot(ff_2[1:fn - 1], abs(f_signal_2[1:fn - 1]))
        ax_2.legend('Y')
        ax_2.plot(pointX_2, pointY_2, color='black')
        for j in range(len(maxTemp_2)):
            ax_2.plot((ff_2[1:fn - 1])[maxTemp_2[j]], abs(f_signal_2[1:fn - 1])[maxTemp_2[j]], 'ro')

        ax_3 = fig.add_subplot(3, 1, 3)
        ax_3.plot(ff_3[1:fn - 1], abs(f_signal_3[1:fn - 1]))
        ax_3.legend('Z')
        ax_3.plot(pointX_3, pointY_3, color='black')
        for j in range(len(maxTemp_3)):
            ax_3.plot((ff_3[1:fn - 1])[maxTemp_3[j]], abs(f_signal_3[1:fn - 1])[maxTemp_3[j]], 'ro')

        ax_3.set_xlabel("Частота (Гц)")
        # plt.savefig(os.path.join(output_path, name+'.png')
        plt.show()
        plt.clf()

    '''
    Objective:
        Функция осуществляет расчет вектора признаков по X,Y,Z характреистикам смектра в диапазоне от 3 до 6 Гц

    Inputs:
        X, Y, Z: массивов с координатами X,Y или Z
        frame: массив с номерами кадров

    Output:
        fmax: значение частоты максимальной гармоники
        ampl: значение амлитуды максимальной гармоники в промежутки от 3 до 6 Гц

    '''

    def _calculate_features(self, X, Y, Z, frame):
        X = np.array(X) - min(X)
        Y = np.array(Y) - min(Y)
        Z = np.array(Z) - min(Z)
        pointY_X, pointX_X, maxTemp_X, f_signal_X, ff_X, fn = self._spectrXYZ(frame, X)
        pointY_Y, pointX_Y, maxTemp_Y, f_signal_Y, ff_Y, fn = self._spectrXYZ(frame, Y)
        pointY_Z, pointX_Z, maxTemp_Z, f_signal_Z, ff_Z, fn = self._spectrXYZ(frame, Z)

        ff36_X = []
        ff36_Y = []
        ff36_Z = []

        for j in range(len(ff_X)-1):
            if ((ff_X[j] > 3) & (ff_X[j] < 6)):
                ff36_X.append(abs(f_signal_X[1:fn - 1])[j])
        for j in range(len(ff_Y)-1):
            if ((ff_Y[j] > 3) & (ff_Y[j] < 6)):
                ff36_Y.append(abs(f_signal_Y[1:fn - 1])[j])
        for j in range(len(ff_Z)-1):
            if ((ff_Z[j] > 3) & (ff_Z[j] < 6)):
                ff36_Z.append(abs(f_signal_Z[1:fn - 1])[j])

        # print(np.where(abs(f_signal_1[1:fn - 1])==max(abs(f_signal_1[1:fn - 1])))[0][0])

        ffmax_X = ff_X[np.where(abs(f_signal_X[1:fn - 1]) == max(abs(f_signal_X[1:fn - 1])))[0][0]]
        ffmax_Y = ff_Y[np.where(abs(f_signal_Y[1:fn - 1]) == max(abs(f_signal_Y[1:fn - 1])))[0][0]]
        ffmax_Z = ff_Z[np.where(abs(f_signal_Z[1:fn - 1]) == max(abs(f_signal_Z[1:fn - 1])))[0][0]]

        f_mean_XYZ, ampl_mean_XYZ = np.mean([ffmax_X, ffmax_Y, ffmax_Z]), np.mean(
            [max(ff36_X), max(ff36_Y), max(ff36_Z)])
        f_max_XYZ, ampl_max_XYZ = np.max([ffmax_X, ffmax_Y, ffmax_Z]), np.max([max(ff36_X), max(ff36_Y), max(ff36_Z)])

        return {'Fmean': f_mean_XYZ, 'Amean': ampl_mean_XYZ, 'Fmax': f_max_XYZ, 'Amax': ampl_max_XYZ}

    def get_features(self, path, file, hand, key_point):
        X, Y, Z, frame = self._tremor_signal(os.path.join(path, file), hand, key_point)
        return self._calculate_features(X, Y, Z, frame)

    def plot_tremor(self, path, file, hand, key_point, output_path):
        X, Y, Z, frame = self._tremor_signal(os.path.join(path, file), hand, key_point)
        self._plot_tremor_signals_and_spectrum(frame, X, Y, Z, file.split('.')[0], output_path)
