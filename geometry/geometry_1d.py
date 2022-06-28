import numpy as np

from .geometry import Geometry
from .sampler import sample
from .. import config


class Interval(Geometry):
    def __init__(self, l, r):
        super().__init__(1, (np.array([l]), np.array([r])), r - l)  #super()是函数是用于调用父类(超类)的一个方法
        self.l, self.r = l, r

    def inside(self, x):
        return np.logical_and(self.l <= x, x <= self.r).flatten()

    def on_boundary(self, x):
        return np.any(np.isclose(x, [self.l, self.r]), axis=-1)   #不明白这里为什么有一个axis = -1

    def distance2boundary(self, x, dirn):  #x到边界的距离
        return x - self.l if dirn < 0 else self.r - x

    def mindist2boundary(self, x):
        return min(np.amin(x - self.l), np.amin(self.r - x))

    def boundary_normal(self, x):
        return -np.isclose(x, self.l).astype(config.real(np)) + np.isclose(x, self.r)  #astype 是数据类型转换,没看懂

    def uniform_points(self, n, boundary=True):
        if boundary:
            return np.linspace(self.l, self.r, num=n, dtype=config.real(np))[:, None]
        return np.linspace(
            self.l, self.r, num=n + 1, endpoint=False, dtype=config.real(np)
        )[1:, None]

    def log_uniform_points(self, n, boundary=True):
        eps = 0 if self.l > 0 else np.finfo(config.real(np)).eps  #不知道这里为什么要有一个eps
        l = np.log(self.l + eps)
        r = np.log(self.r + eps)
        if boundary:
            x = np.linspace(l, r, num=n, dtype=config.real(np))[:, None]
        else:
            x = np.linspace(l, r, num=n + 1, endpoint=False, dtype=config.real(np))[
                1:, None
            ]
        return np.exp(x) - eps

    def random_points(self, n, random="pseudo"):
        x = sample(n, 1, random)
        return (self.diam * x + self.l).astype(config.real(np))

    def uniform_boundary_points(self, n):  #如果n等于1，那就取左端点，如果是大于1的奇数，那就右端点多取一个
        if n == 1:
            return np.array([[self.l]]).astype(config.real(np))
        xl = np.full((n // 2, 1), self.l).astype(config.real(np))
        xr = np.full((n - n // 2, 1), self.r).astype(config.real(np))
        return np.vstack((xl, xr))

    def random_boundary_points(self, n, random="pseudo"):
        if n == 2:
            return np.array([[self.l], [self.r]]).astype(config.real(np))   #如果是两个点，返回左端点和右端点
        return np.random.choice([self.l, self.r], n)[:, None].astype(config.real(np))
#numpy.random.choice(a, size=None, replace=True, p=None)
#从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
#replace:True表示可以取相同数字，False表示不可以取相同数字
#数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。

    def periodic_point(self, x, component=0): #Compute the periodic image of x for periodic boundary condition.
        tmp = np.copy(x)
        tmp[np.isclose(x, self.l)] = self.r
        tmp[np.isclose(x, self.r)] = self.l
        return tmp

    def background_points(self, x, dirn, dist2npt, shift):
        """
        Args:
            dirn: -1 (left), or 1 (right), or 0 (both direction).
            dist2npt: A function which converts distance to the number of extra
                points (not including x).
            shift: The number of shift.
        """

        def background_points_left():
            dx = x[0] - self.l        #与左端点的距离
            n = max(dist2npt(dx), 1)  #额外点数量，最小为1
            h = dx / n                #每个额外点对应的距离
            pts = x[0] - np.arange(-shift, n - shift + 1) * h 
            return pts[:, None]

        def background_points_right():
            dx = self.r - x[0]
            n = max(dist2npt(dx), 1)
            h = dx / n
            pts = x[0] + np.arange(-shift, n - shift + 1) * h
            return pts[:, None]

        return (
            background_points_left()
            if dirn < 0
            else background_points_right()
            if dirn > 0
            else np.vstack((background_points_left(), background_points_right()))
        )
