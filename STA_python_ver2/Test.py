# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 10:57:13 2021

@author: 中南大学自动化学院  智能控制与优化决策课题组
"""

from STA import STA
import Benchmark
import numpy as np
import matplotlib.pyplot as plt

def range_set(Dim):
    Range = np.repeat([0.0, 0.0], Dim).reshape(-1, Dim)
    P1Min = [28] * 24
    P1Max = [200] * 24
    P2Min = [20] * 24
    P2Max = [290] * 24
    P3Min = [30] * 24
    P3Max = [190] * 24
    P4Min = [20] * 24
    P4Max = [260] * 24
    SOCMin = [0.2] * 24
    SOCMax = [0.9] * 24
    PcMin = [0] * 24
    PcMax = [60] * 24
    PdMin = [0] * 24
    PdMax = [60] * 24

    Range[0, :] = np.array([*P1Min, *P2Min, *P3Min, *P4Min, *SOCMin, *PcMin, *PdMin])
    Range[1, :] = np.array([*P1Max, *P2Max, *P3Max, *P4Max, *SOCMax, *PcMax, *PdMax])
    return Range


#参数设置
funfcn = Benchmark.cal_Energy
Dim = 168
Range = range_set(Dim)

SE = 30
Maxiter = 1000
dict_opt = {'alpha':1,'beta':1,'gamma':1,'delta':1}
# 算法调用
sta = STA(funfcn,Range,SE,Maxiter,dict_opt)
sta.run()
# 画图
#x = np.arange(1,Maxiter+2).reshape(-1,1)
y = sta.history
x = np.arange(len(y)).reshape(-1,1)
plt.semilogy(x,y,'b-o')
plt.xlabel('Ierations')
plt.ylabel('fitness')
plt.show()