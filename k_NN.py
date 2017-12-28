import numpy as np

class k_NN:

    # k-近邻 一般k值取比较小的值 交叉验证选取最优的k值
    # 给定一个点和k值， 找出距离该点最近的k个点
    # 根据策略决定该点属于哪一类 如 哪一类的点多 这个点就归为那一类
    # 距离的类型有 欧式距离、Lp距离、Minkowski距离、曼哈顿距离

#############################################################################

    def __init__(self, X):
        # X为训练数据 (d+1)*n d为数据维度 n为样本总数
        # 第d+1行的值为 样本所属的分类类型

        d, n = X.shape
        self.d = d-1
        self.n = n

        l = 0  # 当前的切分坐标轴
        # 生成kd树
        self.root = self.createkdTree(X, l)

#############################################################################

    def getkdTree(self):
        return self.root

##############################################################################

    def getk_NN(self):
        return self.knn

###############################################################################

    def judge(self):
        # 判断分类类型
        cAll = np.array(self.knn)[:, self.d]  # k个近邻的类型
        count = np.bincount(cAll)  # cAll中每个元素出现的次数 下标为元素 值为次数
        c = np.argmax(count)  # 最大值的下标
        return c

###############################################################################

    def midSplit(self, l, X):
        # 将数据按坐标轴l上的中位数 分为两部分 并得到节点
        xl = X[l, :].flatten()  # 当前切分坐标轴下的数据
        n = xl.size
        if np.mod(n, 2) == 0:
            # 为偶数的样本 则去掉第一个 使其为奇数 保证mid为存在的值
            mid = np.ceil(np.median(xl[1:n]))  # 找出中位数
        else:
            mid = np.ceil(np.median(xl))  # 找出中位数

        midIdx = np.where(xl == mid)[0]  # 中位数的下标
        lessIdx = np.where(xl < mid)[0]  # 比中位数小的下标
        greaterIdx = np.where(xl > mid)[0]  # 比中位数大的下标

        node = Node(X[:, midIdx])  # 中位数对应的点作为当前节点
        leftData = X[:, lessIdx]  # 小于中位数的点作为左分支数据
        rightData = X[:, greaterIdx]  # 大于中位数的点作为右分支数据

        return node, leftData, rightData

#################################################################################

    def calculateDistance(self, x1, x2):
        # p >=1
        # p = 2 欧氏距离
        # p = 1 曼哈顿距离
        # p = inf 各个坐标最大距离
        Lp = np.power(np.sum(np.power(np.abs((x1-x2)), self.p)), 1/self.p)
        return Lp

###############################################################################

    def createkdTree(self, data, j):

        # 递归实现
        # 非递归实现要借助 栈
        # node 当前节点
        # data 接下来要分类的左右两个分支
        # j 当前树的深度

        # 计算当前分类基于的维度
        l = np.mod(j, self.d)

        # 根据切分坐标轴 将数据切分 获得当前节点 及左右分支数据
        node, leftData, rightData = self.midSplit(l, data)
        if leftData.size > 0:
            # 左分支有数据 递归
            node.left = self.createkdTree(leftData, j+1)

        if rightData.size > 0:
            # 右分支有数据 递归
            node.right = self.createkdTree(rightData, j+1)

        # 返回当前节点及其子节点
        return node

################################################################################

    def searchkdTree(self, x, p=2, k=1):
        # x为需要分类的点
        # p为距离的类型
        # p >=1
        # p = 2 欧氏距离
        # p = 1 曼哈顿距离
        # p = inf 各个坐标最大距离

        # 也用递归查找吧 不想实现栈先 虽然可以用个数组存储经过的对象
        self.x = x
        self.p = p
        self.k = k  # 要找的k近邻的个数
        self.knn = []  # 距离最近的k个点
        self.distance = []  # 对应k个点的距离

        self.search(self.root)

#############################################################################

    def search(self, node, j = 0):
        # 根据比较当前深度下 获得切分维度
        # 然后 x在该维的值小于节点在该维的值 则左移
        # 大于等于则右移 直到叶节点
        # 然后递归回退 找寻符合条件的点 直到回退到根节点

        # 递归 走到底了
        if node is None:
            return

        # 递归
        # 判断是否是叶节点
        if node.isLeaf():
            # 到了叶节点 计算两点距离 存入距离小的点
            self.updateKNN(node)
            return

        # 递归 直到None或叶节点
        l = np.mod(j, self.d)
        if self.x[l] < node.x[l, 0]:
            # 正常递归
            self.search(node.left, j+1)

            # 回退
            if node.right is not None:
                # 右节点存在 判断该节点是否存在更小的值  若存在 递归查找该分支
                # if self.updateKNN(node.right):  # 这样会导致重复比较 同一个值存入多次
                if self.isUpdate(node.right):  # 改成只判断 不存入 防止重复存入
                    self.search(node.right, j+1)
        else:
            # 正常递归
            self.search(node.right, j+1)

            # 回退
            if node.left is not None:
                # 左节点不为空 判断该节点是否存在更小的值  若存在 递归查找该分支
                # if self.updateKNN(node.left):
                if self.isUpdate(node.left):  # 改成只判断 不存入 防止重复存入
                    self.search(node.left, j+1)

        # 回退 计算当前节点与目标点的距离
        self.updateKNN(node)

####################################################################################

    def updateKNN(self, node):
        # 获取满足条件的点
        n = node.x.shape[1]
        for i in range(n):
            # 计算当前节点中点与待分类点的距离
            xi = node.x[:, i]
            dis = self.calculateDistance(self.x, xi[0:self.d])
            if len(self.distance) < self.k:
                # 还不够k个数据 直接存入
                self.distance.append(dis)
                self.knn.append(xi)
            else:
                # 已经够k个了 找出已有的最大值 与当前值比较
                # 如果当前值较小 则取代该值
                maxDis = np.max(self.distance)
                if maxDis > dis:
                    # 取代当前的最大值
                    maxIdx = np.argmax(self.distance)
                    self.distance[maxIdx] = dis
                    self.knn[maxIdx] = xi

###############################################################################

    def isUpdate(self, node):
        n = node.x.shape[1]
        for i in range(n):
            # 计算当前节点中点与待分类点的距离
            xi = node.x[:, i]
            dis = self.calculateDistance(self.x, xi[0:self.d])
            if len(self.distance) < self.k:
                # 还不够k个数据 直接存入
                return True

            else:
                # 已经够k个了 找出已有的最大值 与当前值比较
                # 如果当前值较小 则取代该值
                maxDis = np.max(self.distance)
                if maxDis > dis:
                    return True
        return False

#############################################################################

# 节点类
class Node:

    def __init__(self, x):
        self.x = x  # 当前节点的数据 可能包含多个样本点

        self.left = None
        self.right = None

    def isLeaf(self):
        return self.right is None and self.left is None

