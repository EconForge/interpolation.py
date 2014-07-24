from __future__ import division

import numpy
import numpy as np

from dolo.numeric.misc import cartesian

class RectangularDomain:

    def __init__(self,smin,smax,orders):
        self.d = len(smin)
        self.smin = smin
        self.smax = smax
        self.bounds = np.row_stack( [smin,smax] )
        self.orders = orders
        nodes = [np.linspace(smin[i], smax[i], orders[i]) for i in range(len(orders))]
        if len(orders) == 1:
            mesh = nodes
        else:
            mesh = np.meshgrid(*nodes)   # works only in 2d
            mesh = [m.T.flatten() for m in mesh]
    #        mesh.reverse()
        self.nodes = nodes
        self.grid = np.row_stack(mesh)

    def find_cell(self, x):
        self.domain = self
        inf = self.smin
        sup = self.smax
        indices = []
        for i in range(self.domain.d):
            xi =(x[i,:] - inf[i])/(sup[i]-inf[i])
            ni = numpy.floor( xi*self.orders[i] )
            ni = numpy.minimum(numpy.maximum(ni,0),self.orders[i]-1)
            indices.append(ni)
        indices = np.row_stack(indices)
        return indices

    def compute_density(self, x):
        cell_indices = self.find_cell(x)
        keep = numpy.isfinite( numpy.sum(cell_indices, axis=1) )
        cell_indices = cell_indices[keep,:]
        npoints = cell_indices.shape[1]
        counts = numpy.zeros(self.orders, dtype=numpy.int)

        #print cell_indices
        for j in range(cell_indices.shape[1]):
            ind = tuple( cell_indices[:,j] )
            counts[ind] += 1
        dens = counts/npoints
        return dens

class TriangulatedDomain:

    def __init__(self,points):
        from scipy.spatial import Delaunay
        self.d = points.shape[0]
        if self.d == 1:
            raise(Exception("Impossible to triangulate in 1 dimension."))
        self.grid = points
        self.delaunay = Delaunay(points.T)
        self.smin = numpy.min(points,axis=1)
        self.smax = numpy.max(points,axis=1)
        self.bounds = np.array( [self.smin,self.smax] )
        


class LinearTriangulation:

    def __init__(self,domain):
        self.domain = domain
        self.delaunay = domain.delaunay
        self.grid = self.domain.grid
        self.smin = domain.smin
        self.smax = domain.smax
        self.bounds = np.row_stack([self.smin, self.smax])

    def __call__(self, zz):
        return self.interpolate(zz)

    def set_values(self, val):
        self.__values__ = val

    def find_simplex(self,points):
        return self.delaunay.find_simplex(points)

    def interpolate(self, points, with_derivatives=False):

        if with_derivatives:
            raise Exception('Option not implemented')

        n_x = self.__values__.shape[0]
        n_p = points.shape[1]
        n_d = self.domain.d

        ndim = self.domain.d
        delaunay = self.delaunay

        points = numpy.minimum(points, self.domain.smax[:,None]) # only for rectangular domains
        points = numpy.maximum(points, self.domain.smin[:,None]) # only for rectangular domains

#        inds_simplices = self.delaunay.find_simplex(points.T)
        inds_simplices = self.find_simplex(points.T)

        inside = (inds_simplices != -1)

        indices = inds_simplices[inside]
        transform = self.delaunay.transform[indices,:,:]
        transform = numpy.rollaxis(transform,0,3)
        vertices = self.delaunay.vertices.T[:,indices]

        Tinv = transform[:ndim,:ndim,:]
        r = transform[ndim,:,:]

        z = points[:,inside]

        from dolo.numeric.serial_operations import serial_dot
        resp = np.zeros((n_x,n_p))

        if with_derivatives:
            dresp = numpy.zeros( (n_x, n_p) )

        all_values_on_vertices = self.__values__[:, vertices]

        for i in range(n_x):
            values_on_vertices = all_values_on_vertices[i,:,:]
            last_V = values_on_vertices[-1,:]
            z_r = z-r
            c = serial_dot(Tinv, z_r)
            D = values_on_vertices[:-1,:] - last_V
            resp[i,inside] = serial_dot(c, D) + last_V

            if with_derivatives:
                interp_dvals = serial_dot(D, Tinv)
                dresp[:,inside] = interp_dvals
                return [resp,dresp]

        else:
            return resp


from numpy import row_stack, column_stack, maximum, minimum, array

class SparseLinear:
    # linear interpolation on a sparse grid

    def __init__(self, smin, smax, l):
        from dolo.numeric.interpolation.smolyak import SmolyakGrid
        sg = SmolyakGrid(smin,smax,l)
        self.smin = array(smin)
        self.smax = array(smax)
        self.bounds = row_stack([smin,smax])
        vertices = cartesian( zip(smin,smax) ).T
        self.grid = column_stack( [sg.grid, vertices] )


#        from scipy.interpolate import LinearNDInterpolator
#        self.interp = LinearNDInterpolator(self.grid.T, (self.grid*0).T)


    def set_values(self, values):

#        if self.interp is None:
        from scipy.interpolate import LinearNDInterpolator
        self.interp = LinearNDInterpolator(self.grid.T, values.T)
#        self.interp.set_values( values.T )
        self.values = values

    def __call__(self, s):
        return self.interpolate(s)

    def interpolate(self, s):
        s = maximum(s, self.smin[:,None])
        s = minimum(s, self.smax[:,None])
        resp = self.interp(s.T).T
        resp = np.atleast_2d(resp)
        return resp



if __name__ =='__main__':

    ## test splines
    from numpy import * 
    beta = 0.96
    bounds = array( [[-1], [1]] )
    orders = array( [10] )
    d = bounds.shape[1]
#
#    smin = bounds[0,:]
#    smax = bounds[1,:]
#    interp = SplineInterpolation(orders, smin, smax)
#    grid = interp.grid
#
#    vals = np.sin(grid)
#    interp.set_values(vals)
#
#    xvec = linspace(-1,1,100)
#    xvec = np.atleast_2d(xvec)
#    yvec = interp(xvec)
#
#
#    from matplotlib.pyplot import *
#    plot(interp.grid.flatten(), vals.flatten(),'o')
#    plot(xvec.flatten(),yvec.flatten())
    #show()


    from dolo.numeric.quantization import standard_quantization_weights
    from matplotlib import pyplot

    f = lambda x: 1 - x[0:1,:]**2 - x[1:2,:]**2

    ndim = 2
    N = 10
    [weights, points] = standard_quantization_weights( N, ndim )

    domain = TriangulatedDomain(points)
    interp = LinearTriangulation(domain)

    values = f(domain.grid)

    interp.set_values(values)


    orders = [100,100]
    smin = domain.smin
    smax = domain.smax
    extent = [smin[0], smax[0], smin[1], smax[1]]
    recdomain = RectangularDomain(smin,smax,orders)
    true_values = f( recdomain.grid )
    linapprox_values = interp(recdomain.grid)

    points = domain.grid

    pyplot.figure()
    pyplot.subplot(221)
#    pyplot.axes().set_aspect('equal')
    pyplot.imshow( true_values.reshape(orders), extent=extent,origin='lower', interpolation='nearest' )

    pyplot.plot(points[0,:],points[1,:],'o')
    pyplot.colorbar()
    pyplot.subplot(222)
    print(linapprox_values.shape)
    pyplot.imshow(linapprox_values.reshape(orders))
    pyplot.colorbar()
    pyplot.subplot(223)
    pyplot.imshow( (true_values- linapprox_values).reshape(orders))
    pyplot.colorbar()


    pyplot.show()
    exit()

    pyplot.figure()




    minbound = delaunay.min_bound
    maxbound = delaunay.max_bound
    extent = [minbound[0],maxbound[0],minbound[1],maxbound[1]]
    pyplot.bone()
    pyplot.figure()
    pyplot.imshow( values, extent=extent, origin='lower' )
    pyplot.colorbar()
    pyplot.figure()
    pyplot.imshow( interp_values, extent=extent,origin='lower', interpolation='nearest' )
    pyplot.axes().set_aspect('equal')
    for el in triangles: #triangulation.get_elements():
        plot_triangle(el)
    pyplot.colorbar()
    pyplot.figure()
    pyplot.imshow( abs(interp_values - values), extent=extent,origin='lower' )
    pylab.colorbar()

    triangles = []
    for v in delaunay.vertices:
        pp = [points[e,:] for e in v]
        triangles.append(pp)
    def plot_triangle(tr):
        ttr = tr + [tr[0]]
        ar = numpy.array(ttr)
        pyplot.plot(ar[:,0],ar[:,1], color='black')
    pyplot.figure()
    pyplot.axes().set_aspect('equal')
    pyplot.plot(points[:,0],points[:,1],'o')
    for el in triangles: #triangulation.get_elements():
        plot_triangle(el)
