from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import argrelmax
from scipy.signal import argrelmin
from pylab import*
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
import matplotlib.pyplot as plt
from scipy import loadtxt, optimize
from scipy.signal import argrelmax
from scipy.signal import argrelmin

'''
реализация алгоритма автоматизированной расстановки точек экстремума для сигналов мимической активности
'''

class AutoFace():
    def __init__(self, **config):
        self.config = config

    def signalProc(self, values, frame):
        filtered = lowess(values, frame, is_sorted=True, frac=0.02, it=0)
        pixel, value = np.array(filtered[:, 0]), np.array(filtered[:, 1])
        maxTemp = argrelmax(value, order=20)
        minTemp = argrelmin(value, order=20)
        values = np.array(values)
        maxP, minP, maxA, minA = list(maxTemp[0]), list(minTemp[0]), list(values[maxTemp[0]]), list(values[minTemp[0]])
        return maxP, minP, maxA, minA

    def _calcMean(self, d):
        n_sumA = []
        n_sumC = []
        n_sumT = []

        for n in d:
            n_sumA.append(n['A'])
            n_sumC.append(n['coord'])
            n_sumT.append(n['type'])
        return {'A': sum(n_sumA) / len(n_sumA), 'coord': round(sum(n_sumC) / len(n_sumC)),
                'type': sum(n_sumT) / len(n_sumT)}
    def _pointcalc(self, nn, n, index):
        nn = np.array(nn)
        # print(nn)
        idx_pairs = np.where(np.diff(np.hstack(([False], nn == index, [False]))))[0].reshape(-1, 2)
        # print(idx_pairs.tolist())
        ind = idx_pairs.tolist()
        minPoint = []
        for i in range(len(ind)):
            minP = []
            for j in range(ind[i][0], ind[i][1]):
                minP.append(n[j])
            if len(minP) > 1:
                minPoint.append(self._calcMean(minP))
            else:
                minPoint.append(minP[0])
        # print(minPoint)
        return minPoint
    def _clearlist2(self, maxPointX, minPointX, maxPointY, minPointY):
        d = []
        nn = []

        for i in range(len(maxPointX)):
            d.append({'A': maxPointY[i], 'coord': maxPointX[i], 'type': 1})
        for i in range(len(minPointX)):
            d.append({'A': minPointY[i], 'coord': minPointX[i], 'type': 0})
        if len(d) != 0:
            n = sorted(d, key=lambda t: t['coord'])

            for i in range(len(n)):
                nn.append(n[i]['type'])
            n_new = []
            minpoint = self._pointcalc(nn, n, 0)
            maxpoint = self._pointcalc(nn, n, 1)
            n_new.extend(minpoint)
            n_new.extend(maxpoint)

            maxPointX1 = []
            minPointX1 = []
            maxPointY1 = []
            minPointY1 = []
            for i in range(len(n_new)):
                if n_new[i]['type'] == 0:
                    minPointX1.append(n_new[i]['coord'])
                    minPointY1.append(n_new[i]['A'])
                if n_new[i]['type'] == 1:
                    maxPointX1.append(n_new[i]['coord'])
                    maxPointY1.append(n_new[i]['A'])

            return maxPointX1, minPointX1, maxPointY1, minPointY1
        else:
            return maxPointX, minPointX, maxPointY, minPointY

    def get_point(self, values, frame):
        #frac, orderMin, orderMax = 0.02, 20, 20
        if (len(values) < 1000):
            frac, orderMin, orderMax = 0.02, 20, 20
        if (len(values) > 1000):
            frac, orderMin, orderMax = 0.03, 30, 30
        filtered = lowess(values, frame, is_sorted=True, frac=frac, it=0)  # 0.02
        pixel, value = np.array(filtered[:, 0]), np.array(filtered[:, 1])
        maxTemp = argrelmax(value, order=orderMax)
        maxes = []

        for maxi in maxTemp[0]:
            maxes.append(maxi)

        pixel, value = np.array(filtered[:, 0]), np.array(filtered[:, 1])
        minTemp = argrelmin(value, order=orderMin)
        mines = []
        for mini in minTemp[0]:
            # if value[mini] < 0:
            mines.append(mini)
        for i in range(len(values) - 1):
            if ((values[i] == 0) & (values[i + 1] > 0)):
                mines.append(i)
            if ((values[i] > 0) & (values[i + 1] == 0)):
                mines.append(i + 1)
        znach = np.array(values)
        maxP, minP, maxA, minA = self._clearlist2(list(maxes), list(mines), list(znach[maxes]), list(znach[mines]))

        datapoint = []
        for i in range(len(maxP)):
            datapoint.append({"Type": 1, "Scale": 1.0, "Brush": "#FFFF0000", "X": maxP[i], "Y": maxA[i]})
        for i in range(len(minP)):
            datapoint.append({"Type": 0, "Scale": 1.0, "Brush": "#FF0000FF", "X": minP[i], "Y": minA[i]})
        if len(datapoint) != 0:
            datapoint = sorted(datapoint, key=lambda t: t['X'])
            if datapoint[-1]['Type'] == 1:
                del datapoint[-1]
            if datapoint[0]['Type'] == 1:
                del datapoint[0]

        maxPointX = []
        minPointX = []
        maxPointY = []
        minPointY = []
        for point in datapoint:
            if point['Type'] == 1:
                maxPointX.append(point['X'])
                maxPointY.append(point['Y'])
            if point['Type'] == 0:
                minPointX.append(point['X'])
                minPointY.append(point['Y'])

        return maxPointX, minPointX, maxPointY, minPointY, frac, orderMin, orderMax