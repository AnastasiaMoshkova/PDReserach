import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import medfilt
from scipy.signal import  hilbert, butter, lfilter

TIMESTAMP_COEFFICIENT = 1000000
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

    def _tremor_signal(self, file, hand_type, key_point, start, stop):
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
                frame.append(data[i][hand]['info']['timestamp'])
        frame = [(fr - frame[0])/TIMESTAMP_COEFFICIENT for fr in frame]
        frame = np.array(frame)

        X = np.array(X)[(frame < stop) & (frame > start)].tolist()
        Y = np.array(Y)[(frame < stop) & (frame > start)].tolist()
        Z = np.array(Z)[(frame < stop) & (frame > start)].tolist()
        frame = frame[(frame < stop) & (frame > start)].tolist()

        df = pd.DataFrame({'X':X, 'Y':Y, 'Z':Z, 'frame':frame})

        if len(X) > 0:
            X = signal.detrend(X)
            Y = signal.detrend(Y)
            Z = signal.detrend(Z)

            k = 21
            smoothed_x = self.meanfilt(X, k)
            smoothed_y = self.meanfilt(Y, k)
            smoothed_z = self.meanfilt(Z, k)

            # smoothed_x = ndimage.median_filter(x, size=20) # median_filter
            # smoothed_y = ndimage.median_filter(y, size=20)
            X = X - smoothed_x
            Y = Y - smoothed_y
            Z = Z - smoothed_z

        return X, Y, Z, frame


    # from https://github.com/peach-lucien/PoET/blob/main/PoET/features.py
    def meanfilt(self, x, k):
        """Apply a length-k mean filter to a 1D array x.
        Boundaries are extended by repeating endpoints.
        """

        assert k % 2 == 1, "Mean filter length must be odd."
        assert x.ndim == 1, "Input must be one-dimensional."

        k2 = (k - 1) // 2
        y = np.zeros((len(x), k), dtype=x.dtype)
        y[:, k2] = x
        for i in range(k2):
            j = k2 - i
            y[j:, i] = x[:-j]
            y[:j, i] = x[0]
            y[:-j, -(i + 1)] = x[j:]
            y[-j:, -(i + 1)] = x[-1]
        return np.mean(y, axis=1)

    # from https://github.com/peach-lucien/PoET/blob/main/PoET/features.py
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        return butter(order, [lowcut, highcut], fs=fs, btype='band')

    # from https://github.com/peach-lucien/PoET/blob/main/PoET/features.py
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    # from https://github.com/peach-lucien/PoET/blob/main/PoET/features.py
    def _spectrum(self, frame, y):

        time = np.array(frame)

        fft_result = np.fft.fft(y)
        frequency = np.fft.fftfreq(y.size, d= time[1] - time[0])

        # Calculate the magnitude of the FFT (amplitude spectrum)
        amplitude = 2 * np.abs(fft_result) / len(
            y)  # Multiplying by 2 because the spectrum is symmetrical for real-valued signals

        # positive frequencies only
        amplitude = amplitude[frequency >= 0]
        frequency = frequency[frequency >= 0]

        return amplitude, frequency

    '''
    Objective:
        Функция осуществляет построение спектра сигнала для 3 координат X,Y,Z на одном графике

    Inputs:
        X, Y, Z: массивов с координатами X,Y или Z
        frame: массив с номерами кадров
        name: название файла

    '''

    def _plot_tremor_signals_and_spectrum(self, frame, X, Y, Z, name, output_path):
        X = np.array(X) - np.min(X)
        Y = np.array(Y) - np.min(Y)
        Z = np.array(Z) - np.min(Z)

        Xfiltered = self.butter_bandpass_filter(X, 3, 12, 1/(frame[1] - frame[0]), order=5)
        Yfiltered = self.butter_bandpass_filter(Y, 3, 12, 1 / (frame[1] - frame[0]), order=5)
        Zfiltered = self.butter_bandpass_filter(Z, 3, 12, 1 / (frame[1] - frame[0]), order=5)

        aX, fX = self._spectrum(frame, Xfiltered)
        aY, fY = self._spectrum(frame, Yfiltered)
        aZ, fZ = self._spectrum(frame, Zfiltered)

        ylim = 0.4
        ylimfrq = 15
        fig = plt.figure(figsize=(15, 11))
        ax_1 = fig.add_subplot(3, 1, 1)
        ax_1.plot(fX, aX)
        ax_1.legend('X')
        ax_1.set_xlim(0, ylimfrq)
        ax_1.set_ylim(0, ylim)

        ax_2 = fig.add_subplot(3, 1, 2)
        ax_2.plot(fY, aY)
        ax_2.legend('Y')
        ax_2.set_xlim(0, ylimfrq)
        ax_2.set_ylim(0, ylim)

        ax_3 = fig.add_subplot(3, 1, 3)
        ax_3.plot(fZ, aZ)
        ax_3.legend('Z')
        ax_3.set_xlim(0, ylimfrq)
        ax_3.set_ylim(0, ylim)

        ax_3.set_xlabel("Частота (Гц)")
        plt.savefig(os.path.join(output_path, name+'spectrum.png'))
        #plt.show()
        plt.clf()

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
        ax_1.plot(np.array(frame), X)
        ax_1.legend('X')
        ax_1.set_ylabel("Амплитуда (мм)")
        ax_1.set_xlim(-1, 20)

        ax_2 = fig.add_subplot(3, 1, 2)
        ax_2.plot(np.array(frame), Y)
        ax_2.legend('Y')
        ax_2.set_ylabel("Амплитуда (мм)")
        ax_2.set_xlim(-1, 20)

        ax_3 = fig.add_subplot(3, 1, 3)
        ax_3.plot(np.array(frame), Z)
        ax_3.legend('Z')
        ax_3.set_xlabel("Время (с)")
        ax_3.set_ylabel("Амплитуда (мм)")
        ax_3.set_xlim(-1, 20)

        plt.savefig(os.path.join(output_path, name + '_signal.png'))
        #plt.show()
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

    def _calculate_features(self, X, Y, Z, frame, f1, f2, quality):

        if ((len(X)==0) | (quality == 0)):
            return {'Fmean': np.NaN,
                    'Amean': np.NaN,
                    'Fmax': np.NaN,
                    'Amax': np.NaN,
                    'PWmean': np.NaN,
                    'PWmax': np.NaN,
                    }

        Xfiltered = self.butter_bandpass_filter(X, f1, f2, 1 / (frame[1] - frame[0]), order=5)
        Yfiltered = self.butter_bandpass_filter(Y, f1, f2, 1 / (frame[1] - frame[0]), order=5)
        Zfiltered = self.butter_bandpass_filter(Z, f1, f2, 1 / (frame[1] - frame[0]), order=5)

        aX, fX = self._spectrum(frame, Xfiltered)
        aY, fY = self._spectrum(frame, Yfiltered)
        aZ, fZ = self._spectrum(frame, Zfiltered)

        fX, fY, fZ = np.array(fX), np.array(fY), np.array(fZ),
        amax_X, amax_Y, amax_Z = np.max(aX), np.max(aY), np.max(aZ)
        ffmax_X, ffmax_Y, ffmax_Z = fX[aX==amax_X][0], fY[aY==amax_Y][0], fZ[aZ==amax_Z][0]

        f_mean_XYZ, ampl_mean_XYZ = (np.mean([ffmax_X, ffmax_Y, ffmax_Z]),
                                     np.mean([amax_X, amax_Y, amax_Z]))
        f_max_XYZ, ampl_max_XYZ = (np.max([ffmax_X, ffmax_Y, ffmax_Z]),
                                   np.max([amax_X, amax_Y, amax_Z]))

        pwX, pwY, pwZ = np.mean(np.array(aX) ** 2), np.mean(np.array(aY) ** 2), np.mean(np.array(aZ) ** 2)
        pw_mean = np.mean([pwX, pwY, pwZ])
        pw_max = np.max([pwX, pwY, pwZ])

        return {'Fmean': f_mean_XYZ,
                'Amean': ampl_mean_XYZ,
                'Fmax': f_max_XYZ,
                'Amax': ampl_max_XYZ,
                'PWmean': pw_mean,
                'PWmax': pw_max}


    def get_features(self, path, file, hand, key_point, frq, feature_type, start, stop):
        X, Y, Z, frame = self._tremor_signal(os.path.join(path, file), hand, key_point, start, stop)
        quality = self._quality_check(X, Y, Z, file)
        features = self._calculate_features(X, Y, Z, frame, frq[0], frq[1], feature_type, quality = quality)
        return features

    def plot_tremor(self, path, file, hand, key_point, output_path):
        X, Y, Z, frame = self._tremor_signal(os.path.join(path, file), hand, key_point, 2, 18)
        if ((len(X)>0) & (len(Y)>0) & (len(Z)>0)):
            self._plot_tremor_signals_and_spectrum(frame, X, Y, Z, file.split('.')[0] + '_' + hand, output_path)


    def _quality_check(self, X, Y, Z, file):

        #patient 32, 95 ?
        #print(file, np.var(X), np.var(Y), np.var(Z))
        xvar, yvar, zvar  = np.var(X), np.var(Y), np.var(Z)

        threshold = 20
        return 0 if ((xvar > threshold) | (yvar > threshold) | (zvar > threshold)) else 1

