import numpy as np


class Perceptron:

    # 感知机 这里实现对偶型 监督学习
    # 求解数据集的一个超平面 将已分类数据集线性分为两类  平面为：wx+b=0
    # 首先 数据集得线性可分
    # 对损失函数 使用梯度下降的方式 求其最小值 得到线性方程的参数解

#######################################################################################

    def __init__(self, XP, XN):
        # XP  # 正数据集 d*nP d为数据维度 nP为正样本个数
        # XN  # 负数据集 d*nN d为数据维度 nN为负样本个数
        X = np.hstack((XP, XN))
        self.X = X  # 数据集 d*n d为数据维度 n为样本个数
        self.XGram = np.dot(X.T, X)  # 样本集的内积

        self.d, self.n = X.shape
        d, self.nP = XP.shape
        d, self.nN = XN.shape

        # 当函数距离大于等于0时，归为1类，否则归为-1类
        self.Y = np.hstack((np.ones(self.nP), -np.ones(self.nN)))

        self.eta = 1  # 梯度下降的步长

        self.w = 0
        self.b = 0

############################################################################################

    def getResult(self):
        return self.w, self.b

##################################################################################################

    def run(self):

        # 根据梯度下降推出
        # 初始值 w = sum(alpha_i*y_i*x_i)
        # alpha = n_i*eta n_i为该值样本点被误判的次数
        alpha = np.zeros(self.n)
        b = 0  # 初始截距 b = sum(alpha_i*y_i)

        #  迭代 直至没有错误分类的点
        while True:

            # 根据当前的平面 找出所有错误分类点的集合
            funcDisP = np.dot(self.XGram, np.array(np.multiply(self.Y, alpha))).T+b  # wx+b  列向量
            err = np.multiply(self.Y, funcDisP.T) # y*(wx+b)  行向量

            errIdx = np.where(err <= 0)[0]  # 被错分类的点集下标

            if errIdx.size < 1:
                # 迭代结束
                break

            r = np.random.randint(errIdx.size)  # 随机取一个下标
            idx = errIdx[r]

            # 梯度下降  更新参数 被错分次数越多的点 说明其离分界面越近 alpha越大
            alpha[idx] = alpha[idx]+self.eta  # alpha_r = alpha_r+eta
            b = b+self.Y[idx]*self.eta  # b = b+y_r*eta

        self.w = np.dot(self.X, np.array(np.multiply(alpha, self.Y)).T)  # 列向量
        self.b = b

