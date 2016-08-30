class LinearBasis:

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.Phi(*args, **kwargs)

    @property
    def B(self):
        return self.Phi(self.nodes)

    def Phi(self, x, orders=None):
        pass

    def eval(self, x, orders=None):
        return self.Phi(x, orders=orders)

    def filter(self,x):
        pass

class CompactLinearBasis(LinearBasis):

    pass
