#
import numpy as np
import matplotlib.pyplot as plt

import Perceptron
import k_NN
import NaiveBayes


def testPerceptron():
    XP = np.array(((2, 1, 1), (2, 2, 2))).T  # 正样本 (3,3).T (4,3).T
    XN = np.array((5, 2, 0), ndmin=2).T  # 负样本 (1,1).T
    perceptron = Perceptron.Perceptron(XP, XN)  # 初始化
    perceptron.run()  # 学习
    w, b = perceptron.getResult()  # 获取结果

    # 将数据及结果以图像展示出来 只展示二维的 更高维的忽略
    X = np.hstack((XP, XN))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[0].flatten(), X[1].flatten())
    x1 = np.arange(0, 5, 0.1)
    x2 = -(w[0]*x1+b)/w[1]
    ax.plot(x1, x2)
    # fig.savefig('perception.png')


# testPerceptron()

def test_k_NN():
    # 数据为第一二行  第三行为分类
    X = np.array(((2, 3, 1), (5, 4, 2), (9, 6, 1), (4, 7, 1),
                  (8, 1, 2), (7, 2, 2))).T

    # 构建kd树
    k_nn = k_NN.k_NN(X)
    tree = k_nn.getkdTree()

    # 查找k个最近的点
    k_nn.searchkdTree([5, 5], k=4)
    neighbor = k_nn.getk_NN()

    # 根据找出的k个点判断类型
    c = k_nn.judge()
    c

# test_k_NN()

def testNaiveBayes():
    # 第二行 1,2,3 对应 S,M,L
    X = np.array(((1, 1), (1, 2), (1, 2), (1, 1), (1, 1),
                  (2, 1), (2, 2), (2, 2), (2, 3), (2, 3),
                  (3, 3), (3, 2), (3, 2), (3, 3), (3, 3))).T
    Y = np.array((-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1))

    # 输入训练数据
    naiveBayes = NaiveBayes.NaiveBayes(X, Y)
    # 判断类型
    out = naiveBayes.judge([2, 0])
    out

testNaiveBayes()
