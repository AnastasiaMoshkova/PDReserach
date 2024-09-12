from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import argrelmax
from scipy.signal import argrelmin
from pylab import*
from scipy.fftpack import rfft, irfft, fftfreq
'''
реализация алгоритма "Adaptive" автоматизированной расстановки точек экстремума для сигналов двигательной активности рук
'''

class Adaptive():
    def __init__(self, **config):
        self.config = config

    def _calc_spectrIndex(self, z, i1, i2):
        s = []
        for i in range(len(z)):
            if (i > i1):
                if (i < i2):
                    s.append(z[i] ** 2)
        return sqrt(sum(s))

    def _calc_spectr(self, z):
        s = []
        for i in range(len(z)):
            s.append(z[i] ** 2)
        return sqrt(sum(s))

    def _point_alg_signal(self, values, frame, frac, order_min, order_max):
        filtered = lowess(values, frame, is_sorted=True, frac=frac, it=0)  # 0.02
        pixel, value = np.array(filtered[:, 0]), np.array(filtered[:, 1])
        maxPoint = argrelmax(value, order=order_max)
        minPoint = argrelmin(value, order=order_min)
        return maxPoint, minPoint

    def _signalPoint(self, x, y, xA, yA):
        d = []
        deleter = []
        for i in range(len(x)):
            d.append({'A': xA[i], 'coord': x[i], 'type': 1})
        for i in range(len(y)):
            d.append({'A': yA[i], 'coord': y[i], 'type': -1})
        n = sorted(d, key=lambda t: t['coord'])
        # проверка что мин первая и последняя
        for i in range(len(n)):
            while n[0]['type'] != -1:
                del n[0]
            while n[-1]['type'] != -1:
                del n[-1]
        # проверка очередности
        ss = 0
        for i in range(1, len(n) - 1):
            ss += n[i]['type']
            if (ss == -2) or (n[i]['type'] == n[i - 1]['type'] and n[i]['type'] == -1):
                deleter.append(i)
                ss = -1
            if (ss == 2) or (n[i]['type'] == n[i - 1]['type'] and n[i]['type'] == 1):
                deleter.append(i)
                ss = 1
        if (len(deleter) != 0):
            deleter.sort(reverse=True)
            # print(deleter)
            # print(n)
            for i in range(len(deleter)):
                del n[deleter[i]]
        x = []
        y = []
        xA = []
        yA = []
        for i in range(len(n)):
            if n[i]['type'] == -1:
                y.append(n[i]['coord'])
                yA.append(n[i]['A'])
            if n[i]['type'] == 1:
                x.append(n[i]['coord'])
                xA.append(n[i]['A'])
        return x, y, xA, yA

    '''
    получения точек максиммумов и минимумов на основе побдора параметров 
    '''

    def get_point(self, values, frame):
        values = np.array(values)
        W = fftfreq(values.size, d=frame[1] - frame[0])
        f_signal = rfft(values)
        lst = list(abs(f_signal[1:300] / 1000))
        ff = list(W[1:300] * 100)
        m = lst.index(max(lst))
        if m > 20:
            index1, index2 = m - 10, m + 10
        else:
            index1, index2 = 0, m + 10

        if ((ff[m] >= 3.9) & (ff[m] < 10)):
            frac, order_min, order_max = 0.01, 10, 10
        if ((ff[m] <= 2.5)):
            frac, order_min, order_max = 0.03, 30, 30
        if ((ff[m] <= 3.9) & (ff[m] > 2.5)):
            frac, order_min, order_max = 0.02, 20, 20
        if ((ff[m] >= 10)):
            frac, order_min, order_max = 0.005, 10, 10
        if ((ff[m] <= 1)):
            frac, order_min, order_max = 0.01, 10, 10

        K = self._calc_spectrIndex(abs(f_signal[1:300] / 1000), index1, index2) / self._calc_spectr(abs(f_signal[1:300] / 1000))
        if ((K < 0.5)):
            frac, order_min, order_max = 0.005, 10, 10

        maxTemp, minTemp = self._point_alg_signal(values, frame, frac=frac, order_min=order_min, order_max=order_max)
        maxP, minP, maxA, minA = self._signalPoint(list(maxTemp[0]), minTemp[0], list(values[maxTemp[0]]), list(values[minTemp[0]]))
        # maxP, maxA, minP, minA=DeleterAmplitude(maxP, maxA, minP, minA)
        return maxP, minP, maxA, minA, frac, order_min, order_max