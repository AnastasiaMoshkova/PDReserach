import pandas as pd
import numpy as np
import os
import json
import math
class Feature():
    def __init__(self, maxPointX, maxPointY, minPointX, minPointY, norm_coef, datapoint, config):
        self.maxPointX = maxPointX
        self.maxPointY = [y/norm_coef for y in maxPointY]
        self.minPointX = minPointX
        self.minPointY = [y/norm_coef for y in minPointY]
        self.norm_coef = norm_coef
        self.datapoint = datapoint
        self.config = config

    def loadfileInterval_hand(self, datapoint, start, stop):
        counter = 0
        maxPointX = []
        minPointX = []
        maxPointY = []
        minPointY = []
        newlist = []
        newlistSortedAll = sorted(datapoint, key=lambda k: k['X'])
        for point in newlistSortedAll:
            if ((point['X'] >= start) & (point['X'] <= stop)):
                newlist.append(point)
        for point in newlist:
            counter = counter + 1
            if point['Type'] == 1:
                if ((counter != 1) and (counter != len(newlist))):
                    maxPointX.append(point['X'])
                    maxPointY.append(point['Y'])
            if point['Type'] == 0:
                minPointX.append(point['X'])
                minPointY.append(point['Y'])
        return maxPointX, maxPointY, minPointX, minPointY

class NumAF(Feature):
    def calc(self):
        return len(self.maxPointY)

class NumA(Feature):
    def calc(self):
        dp = sorted(self.datapoint, key=lambda k: k['X'])[-1]['X']
        if dp < self.config['stop']:
            return round(len(self.maxPointX) * self.config['stop']/dp)
        else:
            return len(self.maxPointY)

class AvgFrq(Feature):
    def calc(self):
        if ((len(self.maxPointX) != 0) | ((len(self.maxPointX) != 1))):
            result = []
            for i in range(len(self.maxPointX) - 1):
                result.append(1 / (self.maxPointX[i + 1] - self.maxPointX[i]))
            return sum(result) / len(result) * self.config['k']
        else:
            return 0

class VarFrq(Feature):
    def calc(self):
        if ((len(self.maxPointX) != 0) | ((len(self.maxPointX) != 1))):
            result = []
            for i in range(len(self.maxPointX) - 1):
                result.append(1 / (self.maxPointX[i + 1] - self.maxPointX[i]))
            return (np.std(result) / np.mean(result)) * self.config['k']
        else:
            return 0

class AvgVopen(Feature):
    def calc(self):
        result = []
        for i in range(len(self.maxPointX)):
            result.append((self.maxPointY[i] - self.minPointY[i]) / (self.maxPointX[i] - self.minPointX[i]))
        if len(result) != 0:
            return sum(result) / len(result)
        else:
            return 0

class AvgVclose(Feature):
    def calc(self):
        result = []
        for i in range(len(self.maxPointX)):
            result.append((self.maxPointY[i] - self.minPointY[i + 1]) / (self.minPointX[i + 1] - self.maxPointX[i]))
        if len(result) != 0:
            return sum(result) / len(result)
        else:
            return 0

class AvgA(Feature):
    def calc(self):
        result = []
        # for i in range(len(minPointX) - 1):
        for i in range(len(self.maxPointY)):
            result.append(self.maxPointY[i] - self.minPointY[i])
            result.append(self.maxPointY[i] - self.minPointY[i + 1])
        if len(result) != 0:
            return sum(result) / len(result)
        else:
            return 0


class VarA(Feature):
    def calc(self):
        result1 = []
        result2 = []
        # for i in range(len(minPointX) - 1):
        for i in range(len(self.maxPointY)):
            result1.append(self.maxPointY[i] - self.minPointY[i])
            result2.append(self.maxPointY[i]-self.minPointY[i+1])
        std1 = (np.std(result1) / np.mean(result1)) * self.config['k']
        std2 = (np.std(result2) / np.mean(result2)) * self.config['k']
        return np.mean([std1, std2])

class VarVopen(Feature):
    def calc(self):
        result = []
        for i in range(len(self.maxPointX)):
            result.append((self.maxPointY[i] - self.minPointY[i]) / (self.maxPointX[i] - self.minPointX[i]))
        if len(result) != 0:
            return (np.std(result) / np.mean(result)) * self.config['k']
        else:
            return 0


class VarVclose(Feature):
    def calc(self):
        result = []
        for i in range(len(self.maxPointX) - 1):
            result.append((self.maxPointY[i] - self.minPointY[i + 1]) / (self.minPointX[i + 1] - self.maxPointX[i]))
        if len(result) != 0:
            return (np.std(result) / np.mean(result)) * self.config['k']
        else:
            return 0

class DecA(Feature):
    def calc(self):
        start1 = self.config['start']
        stop1 = round((max(self.minPointX) - start1) / 4 + start1)
        maxPointX, maxPointY, minPointX, minPointY = self.loadfileInterval_hand(self.datapoint,start1, stop1)
        amplitude1 = AvgA(maxPointX, maxPointY, minPointX, minPointY, self.norm_coef, self.datapoint, self.config).calc()

        start4 = round(3 * (max(self.minPointX) - start1) / 4 + start1)
        stop4 = round(max(self.minPointX)+1)
        maxPointX, maxPointY, minPointX, minPointY = self.loadfileInterval_hand(self.datapoint, start4, stop4)
        amplitude4 = AvgA(maxPointX, maxPointY, minPointX, minPointY, self.norm_coef, self.datapoint, self.config).calc()
        if amplitude1==0:
            return 0
        else:
            return round(amplitude4 / amplitude1 ,4)

class DecV(Feature):
    def calc(self):
        start1 = self.config['start']
        stop1 = (max(self.minPointX)-start1)/4+start1
        maxPointX, maxPointY, minPointX, minPointY = self.loadfileInterval_hand(self.datapoint, start1, stop1)
        speed1 = np.mean([AvgVopen(maxPointX, maxPointY, minPointX, minPointY, self.norm_coef, self.datapoint, self.config).calc(),
                       AvgVclose(maxPointX, maxPointY, minPointX, minPointY, self.norm_coef, self.datapoint, self.config).calc(),])
        start4 = 3*(max(self.minPointX)-start1)/4+start1
        stop4 = max(self.minPointX)
        maxPointX, maxPointY, minPointX, minPointY = self.loadfileInterval_hand(self.datapoint, start4, stop4)
        speed4 = np.mean([AvgVopen(maxPointX, maxPointY, minPointX, minPointY, self.norm_coef, self.datapoint, self.config).calc(),
                          AvgVclose(maxPointX, maxPointY, minPointX, minPointY, self.norm_coef, self.datapoint, self.config).calc(), ])
        if speed1==0:
            return 0
        else:
            return round(speed4 / speed1 ,4)

class DecFA(Feature):
    def amplitudeDec(self, maxPointX, maxPointY, minPointX, minPointY):
        result = []
        for i in range(len(minPointX) - 1):
            result.append(round(np.mean([maxPointY[i] - minPointY[i], maxPointY[i] - minPointY[i + 1]]), 2))
        if result != 0:
            return result
        else:
            return 0
    def calc(self):
        dec = []
        answ1 = []
        vect = self.amplitudeDec(self.maxPointX, self.maxPointY, self.minPointX, self.minPointY)
        for i in range(len(vect) - 1):
            dec.append(round(vect[i + 1] / vect[i], 2))
            # answ1.append(round(1-(vect[i + 1] / vect[i]), 2))
        counter = 0
        z = []
        for j in range(len(dec)):
            if dec[j] < 1:
                counter = counter + 1
                z.append(1 - dec[j])
        if z != 0:
            return sum(z)
        else:
            return 0

class DecFV(Feature):
    def SpeedDec(self, maxPointX, maxPointY, minPointX, minPointY):
        result = []
        for i in range(len(minPointX) - 1):
            result.append(round(np.mean([(maxPointY[i] - minPointY[i + 1]) / (minPointX[i + 1] - maxPointX[i]),
                                         (maxPointY[i] - minPointY[i]) / (maxPointX[i] - minPointX[i])]), 2))
        if result != 0:
            return result
        else:
            return 0

    def calc(self):
        dec = []
        answ1 = []
        vect = self.SpeedDec(self.maxPointX, self.maxPointY, self.minPointX, self.minPointY)
        for i in range(len(vect) - 1):
            dec.append(round(vect[i + 1] / vect[i], 2))
            # answ1.append(round(1-(vect[i + 1] / vect[i]), 2))
        counter = 0
        z = []
        for j in range(len(dec)):
            if dec[j] < 1:
                counter = counter + 1
                z.append(1 - dec[j])
        if z != 0:
            return sum(z)
        else:
            return 0

class Length(Feature):
    def calc(self):
        if (len(self.maxPointX) == 10):
            return (self.minPointX[-1] - self.minPointX[0])
        else:
            length = (self.minPointX[-1] - self.minPointX[0]) * 10 / len(self.maxPointX)
            return length


class AvgFrqF(Feature):
    def calc(self):
        if ((len(self.maxPointX) != 0) | ((len(self.maxPointX) != 1))):
            result = []
            for i in range(len(self.maxPointX) - 1):
                result.append(self.config['k'] / (self.maxPointX[i + 1] - self.maxPointX[i]))
            return sum(result) / len(result)
        else:
            return 0

class VarFrqF(Feature):
    def calc(self):
        if ((len(self.maxPointX) != 0) | ((len(self.maxPointX) != 1))):
            result = []
            for i in range(len(self.maxPointX) - 1):
                result.append(self.config['k'] / (self.maxPointX[i + 1] - self.maxPointX[i]))
            return (np.std(result) / np.mean(result)) * self.config['k']
        else:
            return 0


class DecValueA(Feature):
    def amplitudeDec(self, maxPointX, maxPointY, minPointX, minPointY):
        result = []
        for i in range(len(minPointX) - 1):
            result.append(round(np.mean([maxPointY[i] - minPointY[i], maxPointY[i] - minPointY[i + 1]]), 2))
        return result
    def calc(self):
        dec=[]
        answ1=[]
        vect = self.amplitudeDec(self.maxPointX, self.maxPointY, self.minPointX, self.minPointY)
        for i in range(len(vect)-1):
            dec.append(round(vect[i+1]/vect[i],2))
            #answ1.append(round(1-(vect[i + 1] / vect[i]), 2))
        counter = 0
        z = []
        for j in range(len(dec)):
            if dec[j] < 1:
                counter = counter + 1
                z.append(1 - dec[j])
        return sum(z)


class decValueV(Feature):
    def speedDec(self,maxPointX, maxPointY, minPointX, minPointY):
        result = []
        for i in range(len(minPointX) - 1):
            # print((maxPointY[i]-minPointY[i+1])/(minPointX[i+1]-maxPointX[i]))
            # print((maxPointY[i]-minPointY[i])/(maxPointX[i]-minPointX[i]))
            result.append(round(np.mean([(maxPointY[i] - minPointY[i + 1]) / (minPointX[i + 1] - maxPointX[i]),
                                         (maxPointY[i] - minPointY[i]) / (maxPointX[i] - minPointX[i])]), 2))
        # print(result)
        return result
    def calc(self):
        dec=[]
        answ1=[]
        vect = self.speedDec(self.maxPointX, self.maxPointY, self.minPointX, self.minPointY)
        for i in range(len(vect)-1):
            dec.append(round(vect[i+1]/vect[i],2))
            #answ1.append(round(1-(vect[i + 1] / vect[i]), 2))
        counter = 0
        z = []
        for j in range(len(dec)):
            if dec[j] < 1:
                counter = counter + 1
                z.append(1 - dec[j])
        return sum(z)