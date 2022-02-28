#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import reduce
from operator import mul
import numpy as np
import math


def Sphere(s):
	square = map(lambda y: y * y ,s)
	return sum(square)

def Rosenbrock(s):
	square = map(lambda x, y: 100 * (y - x**2)**2 + (x - 1)**2 , s[:len(s)-1],s[1:len(s)])
	return  sum(square)

def Rastrigin(s):
	square = map(lambda x: x**2 - 10 * math.cos(2*math.pi*x) + 10, s)
	return  float(sum(square))

def Michalewicz(s):
	n = len(s)
	t1 = -sum(map(lambda x, y: math.sin(x) * (math.sin( y*x**2/math.pi ))**20, s, range(1,n+1)))
	return t1

def Griewank(s): 
	t1 = sum(map(lambda x: 1/4000 * x**2, s))
	n = len(s)
	t2 = map(lambda x, y: math.cos(x/np.sqrt(y)), s, range(1,n+1))
	t3 = reduce(mul, t2)
	return t1 - t3 + 1


def cal_Energy(X,mk = 100):  # m(k)：惩罚因子，随迭代次数 k 逐渐增大
	# 定义优化问题的参数
	Cap = 100  # 电池容量
	SOC_ini = 0.3  # 初始SOC
	SOC_exp = 0.8  # 终止SOC
	eta_c = 0.95  # 电池放电效率
	eta_d = 0.9  # 电池充电功率

	N_g = 4  # 发电机组数及调度时长
	N_t = 24

	a = np.array([0.12, 0.17, 0.15, 0.19])  # 四组火力发电机的对应成本系数
	b = np.array([14.8, 16.57, 15.55, 16.21])
	c = np.array([89, 83, 100, 70])

	Pg_min = np.array([28, 20, 30, 20])  # 四组火力发电机的对应发电上下限
	Pg_max = np.array([200, 290, 190, 260])

	RU = np.array([40, 30, 30, 50])  # 四组火力发电机的爬坡约束
	RD = np.array([40, 30, 30, 50])

	Load = np.array([510, 530, 516, 510, 515, 544, 646, 686, 741, 734, 748, 760, \
					 754, 700, 686, 720, 714, 761, 727, 714, 618, 584, 578, 544])  # 24小时对应负荷
	# 定义优化问题的决策变量
	P = np.resize(X[0:96], (4, 24))
	SOC = X[24 * 4:24 * 5]
	Pd = X[24 * 5:24 * 6]
	Pc = X[24 * 6:24 * 7]

	# 给出目标函数
	fx = sum([a[g] * P[g, t] * P[g, t] + b[g] * P[g, t] + c[g] for t in range(N_t) for g in range(N_g)])
	# fx = sum(a[g]*P[g, t]*P[g, t]+b[g]*P[g, t]+c[g] for t in range(N_t) for g in range(N_g))

	# 给出罚函数
	# a = np.array(sum([max(0,P[g,t]-Pg_max[g])for t in range(N_t) for g in range(N_g)]))
	p1 = sum([max(0, P[g, t] - Pg_max[g]) for t in range(N_t) for g in range(N_g)])  # 发电上限
	p2 = sum([max(0, -P[g, t] + Pg_min[g]) for t in range(N_t) for g in range(N_g)])  # 发电下限
	p3 = sum([max(0, P[g, t + 1] - P[g, t] - RU[g]) for t in range(N_t - 1) for g in range(N_g)])  # 爬坡上限
	p4 = sum([max(0, P[g, t] - P[g, t + 1] - RD[g]) for t in range(N_t - 1) for g in range(N_g)])  # 爬坡下限
	p5 = sum([max(0, Load[t] + Pc[t] - Pd[t] - sum([P[g][t] for g in range(N_g)])) for t in
			  range(N_t)])  # 总的发电功率加放电功率≥负荷加充电功率
	p6 = np.abs(Cap * SOC[0] - Cap * SOC_ini - Pc[0] * eta_c + Pd[0] / eta_d)  # 电池初始SOC状态
	p7 = sum(np.abs(
		[Cap * SOC[t] - Cap * SOC[t - 1] - Pc[t] * eta_c + Pd[t] / eta_d for t in range(1, N_t - 1)]))  # 电池SOC状态更新
	p8 = np.abs(Cap * SOC[23] - Cap * SOC_exp)  # 电池终止SOC状态
	# return fx+mk*p1
	p9 = sum([max(0, SOC[t] - 0.9) for t in range(N_t)])  # SOC上限
	p10 = sum([max(0, 0.2 - SOC[t]) for t in range(N_t)])  # SOC下限
	p11 = sum([max(0, Pc[t] - 60) for t in range(N_t)])  # Pc上限
	p12 = sum([max(0, -Pc[t]) for t in range(N_t)])  # Pc下限
	p13 = sum([max(0, Pd[t] - 60) for t in range(N_t)])  # Pd上限
	p14 = sum([max(0, -Pd[t]) for t in range(N_t)])  # Pd上限
	# print('{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14))
	return fx + mk * (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 + p11 + p12 + p13 + p14)


