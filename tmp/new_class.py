from interpolation.smolyak import *

class Smolyak:

    def __init__(self, a, b, l, values=None):

        d = len(a)
        self.__sg__ = SmolyakGrid(d, l, lb=a, ub=b)
        self.grid = self.__sg__.grid

        if values is not None:
            self.set_values(values)

    def set_values(self, x):

        self.__sinterp__ = SmolyakInterp(self.__sg__, x)

    def interpolate(self, s, deriv=False, deriv_X=False):

        res = self.__sinterp__.interpolate(s, deriv=deriv, deriv_X=deriv_X)

        return res

class MultiSmolyak:

    def __init__(self, a, b, l, values=None):

        d = len(a)
        self.d = d
        self.__sg__ = SmolyakGrid(d, l, lb=a, ub=b)
        self.grid = self.__sg__.grid

        if values is not None:
            self.set_values(values)

    def set_values(self, x):

        self.n_x = x.shape[1]
        self.__values__ = x
        self.__sinterp__ = [SmolyakInterp(self.__sg__, xi) for xi in x.T]

    def interpolate(self, s, deriv=False, deriv_X=False):

        from numpy import zeros
        N = s.shape[0]

        res = zeros((N, self.n_x))
        l = [res]
        if deriv:
            res_s = zeros((N, self.n_x, self.d))
            resl.append(res_s)
        if deriv_X:
            T = self.__values__.shape[0]
            res_x = zeros((N, self.n_x, T))
            l.append(res_x)
        for i,sinterp in enumerate(self.__sinterp__):
            l = sinterp.interpolate(s, deriv=deriv,deriv_X=deriv_X)
            if not (deriv or deriv_X):
                res[:,i] = l
            else:
                res[:,i] = l[0]
            if deriv:
                res_s[:,i,:] = l[1]
            if deriv_X:
                if not deriv:
                    res_x[:,i,:] = l[1].T
                else:
                    res_x[:,i,:] = l[2].T
        if not (deriv or deriv_X):
            return res
        else:
            return l
        # if not (deriv or deriv_X):
        #     return res
        # l = [res]
        # if deriv:
        #     l.append(res_s)
        # if deriv_X:
        #     l.append(res_x)
        # return l
