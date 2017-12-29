
import numpy as np

class NaiveBayes:
    # 朴素贝叶斯法 假设 不同特征条件概率分布独立（特征，即分类属性）
    # 则有 P(X|Y) = P(X1|Y)*...*P(Xl|Y)  这里Xl表示的是同一个输入的不同特征
    # 分类策略 最大后验概率 即最小期望风险

    def __init__(self, X, Y):
        # X 训练数据 d*n d为数据维度 对应不同特征 n为样本数
        # Y 1*n 每个样本的分类类型
        self.X = X
        self.Y = Y

        self.d, self.n = self.X.shape

        self._calProb()

    # 计算先验概率和条件概率
    def _calProb(self):

        # 找出有那几种分类类型
        classesX = []
        for l in range(self.d):
            classesX.append(np.unique(self.X[l, :]))
        classesY = np.unique(self.Y)
        classesYNum = len(classesY)

        # 计算先验概率和条件概率
        pY = np.zeros((classesYNum))
        # 为 (d*classesYNum)行的list 一行为一个特征的所有可能取值在某种分类下的条件概率
        p_Xjl_Yk = []
        for k in range(classesYNum):  # classesYNum种Y的取值
            idx = np.where(self.Y == classesY[k])[0]
            # 计算先验概率p(Y)
            pY[k] = len(idx)/self.n

            # 计算# 计算条件概率p(Xjl|Yk) = p(Xjl|Yk) jl为Xl的第j种取值
            for l in range(self.d): # d个特征
                # 第l个特征的不同取值的条件概率
                p_Xjl_Yk.append(self._classify(self.X[l, idx], classesX[l]))

        self.classesX = classesX
        self.classesY = classesY
        self.pY = pY
        self.p_Xjl_Yk = p_Xjl_Yk

    # 找出每个属性有几种取值 及相应取值对应的条件概率
    def _classify(self, data, classes):
        # data 1*n 将n个数值按不同取值分类 并计算每个数值的个数
        # classes 1*Sj data所有的可能取值
        # out 输出的结果 1*Sj 与classes相对应值的概率 p(Xjl|Yk) = p(Xjl|Yk)
        cNum = len(classes)
        out = np.zeros((cNum))
        for j in range(cNum):
            idx = np.where(self.Y == classes[j])[0]
            out[j] = len(idx)/cNum
        return out

    # 输入一个数据 d*1 判断其所属类别
    def judge(self, x):

        # 找出输入x每个特征取值对应的取值下标
        ids = np.zeros((self.d), dtype='int')
        for l in range(self.d):
            ids[l] = np.where(self.classesX[l] == x[l])[0]

        # 计算每一种分类类型下的后验概率
        classesYNum = len(self.classesY)
        pYX = np.zeros((classesYNum))

        for k in range(classesYNum):
            pYX[k] = self.pY[k]
            offset = k*classesYNum

            # P(X|Y) = P(X1|Y)*...*P(Xl|Y)
            for l in range(self.d):
                pYX[k] = pYX[k] * self.p_Xjl_Yk[offset+l][ids[l]]

        # 找出后验概率最大的值 作为输入的分类类型
        idxMax = np.argmax(pYX)
        return self.classesY[idxMax]
