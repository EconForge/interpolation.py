from __future__ import division

def B0(u, i, x):

    if (u[i] <= x < u[i + 1]):
        return 1.0
    else:
        return 0.0

def dB0(u, i, x):

    return 0.0


def B(p, u, i, x):

    if (p == 0):
        return B0(u, i, x)
    else:
        return (((x-u[i])/(u[i+p]-u[i]))*B(p-1,u,i,x) + ((u[i+p+1]-x)/(u[i+p+1]-u[i+1]))*B(p-1,u,i+1,x))

def dB(p, u, i, x):

    if p==0:
        return 0
    else:
        return ((1.0/(u[i+p]-u[i]))*B(p-1,u,i,x) + \
        ((-1.0)/(u[i+p+1]-u[i+1]))*B(p-1,u,i+1,x)) + \
        (((x-u[i])/(u[i+p]-u[i]))*dB(p-1,u,i,x) + \
        ((u[i+p+1]-x)/(u[i+p+1]-u[i+1]))*dB(p-1,u,i+1,x))

def d2B(p, u, i, x):

    if p==0:
        return 0
    else:
        t = 2*((1.0)/(u[i+p]-u[i]))*dB(p-1,u,i,x) + \
        2*((-1.0)/(u[i+p+1]-u[i+1]))*dB(p-1,u,i+1,x) + \
        (((x-u[i])/(u[i+p]-u[i]))*d2B(p-1,u,i,x) + \
        ((u[i+p+1]-x)/(u[i+p+1]-u[i+1]))*d2B(p-1,u,i+1,x))
        return t

def d3B(p, u, i, x):

    if p==0:
        return 0
    else:
        t = 3*((1.0)/(u[i+p]-u[i]))*d2B(p-1,u,i,x) + \
        3*((-1.0)/(u[i+p+1]-u[i+1]))*d2B(p-1,u,i+1,x) + \
        (((x-u[i])/(u[i+p]-u[i]))*d3B(p-1,u,i,x) + \
        ((u[i+p+1]-x)/(u[i+p+1]-u[i+1]))*d3B(p-1,u,i+1,x))
        return t

from numpy import *

xvec = linspace(-5, 5, 11)

tvec = linspace(-5, 5, 1000)

yvec = array([B0(xvec, 5, x) for x in tvec])
y1vec = array([B(1, xvec, 5, x) for x in tvec])
y2vec = array([B(2, xvec, 5, x) for x in tvec])
y3vec = array([B(3, xvec, 5, x) for x in tvec])


print(B(3, xvec, 5, 0))
print(B(3, xvec, 5, 1))
print(B(3, xvec, 5, 2))
print(B(3, xvec, 5, 3))
print(B(3, xvec, 5, 4))

dyvec = array([dB(0, xvec, 5, x) for x in tvec])
dy1vec = array([dB(1, xvec, 5, x) for x in tvec])
dy2vec = array([dB(2, xvec, 5, x) for x in tvec])
dy3vec = array([dB(3, xvec, 5, x) for x in tvec])


d2yvec = array([d2B(0, xvec, 5, x) for x in tvec])
d2y1vec = array([d2B(1, xvec, 5, x) for x in tvec])
d2y2vec = array([d2B(2, xvec, 5, x) for x in tvec])
d2y3vec = array([d2B(3, xvec, 5, x) for x in tvec])

d3yvec = array([d3B(0, xvec, 5, x) for x in tvec])
d3y1vec = array([d3B(1, xvec, 5, x) for x in tvec])
d3y2vec = array([d3B(2, xvec, 5, x) for x in tvec])
d3y3vec = array([d3B(3, xvec, 5, x) for x in tvec])

print( ([d3B(3, xvec, 5, x) for x in [0.0,1.0, 2.0, 3.0] ]) )

points = [3.0,2.0,1.0,0.0]
tab = [
[d3B(3, xvec, 5, x)/6 for x in points ],
[d2B(3, xvec, 5, x)/2 for x in points ],
[dB(3, xvec, 5, x) for x in points ],
[B(3, xvec, 5, x) for x in points ]
]

from eval_cubic_splines import Ad

tab = (array(tab)).T

print(tab)
print(Ad)

from matplotlib.pyplot import *

subplot(411)
plot(xvec, xvec * 0, 'o-')
plot(tvec, yvec)
plot(tvec, y1vec)
plot(tvec, y2vec)
plot(tvec, y3vec)

subplot(412)
plot(xvec, xvec * 0, 'o-')
plot(tvec, dyvec)
plot(tvec, dy1vec)

plot(tvec, dy2vec)

plot(tvec, dy3vec)


subplot(413)
plot(xvec, xvec * 0, 'o-')
plot(tvec, d2yvec)
plot(tvec, d2y1vec)
plot(tvec, d2y2vec)

plot(tvec, d2y3vec)

subplot(414)
plot(xvec, xvec * 0, 'o-')
plot(tvec, d3yvec)
plot(tvec, d3y1vec)
plot(tvec, d3y2vec)

plot(tvec, d3y3vec)

show()
