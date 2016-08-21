from dprint import dprint

def f(x):
    y = x**2
    dprint( y )
    z = y + 1
    return z

f(1)
