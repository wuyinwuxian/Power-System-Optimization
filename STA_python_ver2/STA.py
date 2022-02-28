# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 10:57:13 2021

@author: 中南大学自动化学院  智能控制与优化决策课题组
"""

import numpy as np
import time

class STA():
    # 初始化
    def __init__(self,funfcn,Range,SE,Maxiter,dict_opt):
        self.funfcn = funfcn
        self.Range = Range
        self.SE = SE
        self.Maxiter = Maxiter
        self.dict_opt = dict_opt
        self.Dim = self.Range.shape[1]

    def initialization(self):
        self.Best = np.array(self.Range[0,:] + self.Range[1,:]-self.Range[1,:]*np.random.uniform(0,1,(1,self.Range.shape[1])))
        self.fBest = self.funfcn(self.Best[0])
        self.history = []
        self.history.append(self.fBest)
        
    """利用旋转变换公式产生新解，物理上是对 Best 为中心 alpha 为半径的超球体内进行采样SE个解"""
    def op_rotate(self):
        R1 = np.random.uniform(-1,1,(self.SE,self.Dim))
        R2 = np.random.uniform(-1,1,(self.SE,1))
        a = np.tile(self.Best,(self.SE,1))
        b = np.tile(R2,(1,self.Dim))
        c = R1/np.tile(np.linalg.norm(R1,axis=1,keepdims = True),(1,self.Dim))
        State = a + self.dict_opt['alpha']*b*c
        return State

    """利用伸缩变换公式产生新解，物理上是对 Best 的  每个  维度都有可能伸缩到（-∞，+∞），然后进行约束到定义域边界，属于全局搜索"""
    def op_expand(self): 
        a = np.tile(self.Best,(self.SE,1))
        b = np.random.randn(self.SE,self.Dim)
        State = a + self.dict_opt['gamma'] * b * a
        return State

    """利用轴向变换公式产生新解，物理上是对 Best 的  单个 维度都有可能伸缩到（-∞，+∞），属于局部搜索，增强的是单维搜索能力"""
    def op_axes(self):
        #实现一
        State = np.tile(self.Best,(self.SE,1))
        for i in range(self.SE):
            index = np.random.randint(0,self.Dim)
            State[i,index] = State[i,index] + self.dict_opt['delta']*np.random.randn()*State[i,index]
        return State
        #实现二
#        a = np.tile(self.Best,(self.SE,1))
#        A = np.zeros((self.SE,self.Dim))
#        A[np.arange(self.SE),np.random.randint(0,self.Dim,self.SE)] = np.random.randn(self.SE)
#        State = a + A*a
#        return State
    
    """利用平移公式产生新解，物理上是在 oldBest —— newBest 这条直线上进行搜索产生新解，我们认为新旧两个最优解的连线上大概率会出现好的解，比如两个解在谷底两侧时"""
    def op_translate(self,oldBest):
        a = np.tile(self.Best,(self.SE,1)) # SE * n
        b = np.random.uniform(0,1,(self.SE,1))
        c = self.dict_opt['beta']*(self.Best - oldBest)/(np.linalg.norm(self.Best - oldBest)) # 1*n
        State = a + b * c
        return State

    def selection(self,State):
        fState = np.zeros((self.SE,1)) #
        for i in range(self.SE):
            fState[i] = self.funfcn(State[i,:])
        index = np.argmin(fState)
        return State[index,:],fState[index,:]
    
    def bound(self,State):
        Pop_Lb = np.tile(self.Range[0],(State.shape[0],1))
        Pop_Ub = np.tile(self.Range[1],(State.shape[0],1))
        changeRows = State > Pop_Ub
        State[changeRows] = Pop_Ub[changeRows]
        changeRows = State < Pop_Lb
        State[changeRows] = Pop_Lb[changeRows]
        return State       

    def run(self):
        time_start = time.time()
        self.initialization()
        for i in range(self.Maxiter):
            if self.dict_opt['alpha'] < 1e-4:
                self.dict_opt['alpha'] = 1.0
            else:
                self.dict_opt['alpha'] = 0.5*self.dict_opt['alpha']

            """循环使用三个算子产生新解并且更新最优解"""
            dict_op = {"rotate": self.op_rotate(), "expand": self.op_expand(),"axes": self.op_axes()}
            for key in dict_op:
                oldBest = self.Best
                State = self.bound(dict_op[key])
                newBest, fnewBest = self.selection(State)
                if fnewBest < self.fBest:      # 如果算子（三个中任意一个）产生得解确实比历史得好，那再调用translate算子试试能不能找到更好得
                    self.fBest, self.Best = fnewBest, newBest
                    State = self.bound(self.op_translate(oldBest))
                    newBest, fnewBest = self.selection(State)
                    if fnewBest < self.fBest:
                        self.fBest, self.Best = fnewBest, newBest

            self.history.append(self.fBest[0])
            print("第{}次迭代的最优值是:{:.5e}".format(i,self.fBest[0]))

            # """让算法提前停止,如果两次最优值的更新量小于 1e-10"""
            # if (abs(self.history[i+1] - self.history[i]) < 1e-10) & (self.dict_opt['alpha'] < 1e-4):
            #     self.history = np.array(self.history[1:])
            #     break
        time_end = time.time()
        print("总耗时为:{:.5f}".format(time_end - time_start))
