def B0(u, i, x):

    if u[i] <= x < u[i + 1]:
        return 1.0
    else:
        return 0.0


def B(p, u, i, x):

    if p == 0:
        return B0(u, i, x)
    else:
        return ((x - u[i]) / (u[i + p] - u[i])) * B(p - 1, u, i, x) + (
            (u[i + p + 1] - x) / (u[i + p + 1] - u[i + 1])
        ) * B(p - 1, u, i + 1, x)


#
# from pylab import *
#
# xvec = linspace(-5, 5, 11)
#
# tvec = linspace(-5, 5, 1000)
#
# yvec = array([B0(xvec, 5, x) for x in tvec])
# y2vec = array([B(2, xvec, 5, x) for x in tvec])
# y3vec = array([B(3, xvec, 5, x) for x in tvec])
#
#
# print(B(3, xvec, 5, 0))
# print(B(3, xvec, 5, 1))
# print(B(3, xvec, 5, 2))
# print(B(3, xvec, 5, 3))
# print(B(3, xvec, 5, 4))
#
#
# plot(xvec, xvec * 0, 'o-')
#
# plot(tvec, yvec)
# plot(tvec, y2vec)
# plot(tvec, y3vec)
#
#
# show()
