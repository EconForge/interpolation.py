import tempita
from functools import reduce
import operator
from numba import generated_jit
import numpy as np

def print_expr(symbs, inds=[], multispline=False):
    if len(symbs) == 0:
        if multispline:
            return 'coefs[{},k]'.format(str.join(',',['i_{}+{}'.format(i,k) for i,k in enumerate(inds)]))
        else:
            return 'coefs[{}]'.format(str.join(',',['i_{}+{}'.format(i,k) for i,k in enumerate(inds)]))
    else:
        h = symbs[0]
        q = symbs[1:]
        exprs = [  '{}_{}*({})'.format(h,i,print_expr(q,inds + [i], multispline=multispline)) for i in range(4)]
        return str.join( ' + ', exprs )


template = '''
import numpy as np
def kronecker_times_compact_{{d}}d(matrices, coefs):
    {{for i in range(d)}}
    (ind_{{i}}, vals_{{i}}, m_{{i}}) = matrices[{{i}}]
    {{endfor}}
    N = vals_0.shape[0]
    n_x = coefs.shape[{{d}}]
    res = np.zeros((N, n_x))
    for n in range(N):
        {{for i in range(d)}}
        i_{{i}} = ind_{{i}}[n]
        Phi_{{i}}_0 = vals_{{i}}[n,0]
        Phi_{{i}}_1 = vals_{{i}}[n,1]
        Phi_{{i}}_2 = vals_{{i}}[n,2]
        Phi_{{i}}_3 = vals_{{i}}[n,3]
        {{endfor}}
        for k in range(n_x):
            res[n,k] = {{print_expr(['Phi_'+str(i) for i in range(d)], multispline=True) }}
    return res
    '''

def Phi_d_Phi_vec(n,i):
    l = ['Phi_'+str(j) for j in range(n)]
    l[i] = 'd_Phi_'+str(i)
    return l


template_diff = '''
import numpy as np
def kronecker_times_compact_diff_{{d}}d(matrices, coefs):
    {{for i in range(d)}}
    (ind_{{i}}, vals_{{i}}, m_{{i}}) = matrices[{{i}}]
    {{endfor}}
    N = vals_0.shape[0]
    n_x = coefs.shape[{{d}}]
    res = np.zeros((N, n_x, {{d+1}}))
    for n in range(N):
        {{for i in range(d)}}
        i_{{i}} = ind_{{i}}[n]
        Phi_{{i}}_0 = vals_{{i}}[n,0,0]
        Phi_{{i}}_1 = vals_{{i}}[n,1,0]
        Phi_{{i}}_2 = vals_{{i}}[n,2,0]
        Phi_{{i}}_3 = vals_{{i}}[n,3,0]
        d_Phi_{{i}}_0 = vals_{{i}}[n,0,1]
        d_Phi_{{i}}_1 = vals_{{i}}[n,1,1]
        d_Phi_{{i}}_2 = vals_{{i}}[n,2,1]
        d_Phi_{{i}}_3 = vals_{{i}}[n,3,1]
        {{endfor}}
        for k in range(n_x):
            res[n,k,0] = {{print_expr(['Phi_'+str(i) for i in range(d)], multispline=True) }}
            {{for i_d in range(d)}}
            res[n,k,{{i_d+1}}] = {{print_expr(Phi_d_Phi_vec(d,i_d), multispline=True)}}
            {{endfor}}

    return res
    '''


@generated_jit
def kronecker_times_compact(matrices, coefs):
    d = len(matrices)
    tmpl = tempita.Template(template)
    source = tmpl.substitute(d=d, print_expr=print_expr)
    res = exec(source)
    context = {}
    comp_source = compile(source,'<string>','exec')
    res = eval(comp_source, context)
    key = 'kronecker_times_compact_{}d'.format(d)
    fun = context[key]
    return fun

@generated_jit
def kronecker_times_compact_diff(matrices, coefs):
    d = len(matrices)
    tmpl = tempita.Template(template_diff)
    source = tmpl.substitute(d=d, print_expr=print_expr, Phi_d_Phi_vec=Phi_d_Phi_vec)
    res = exec(source)
    context = {}
    comp_source = compile(source,'<string>','exec')
    res = eval(comp_source, context)
    key = 'kronecker_times_compact_diff_{}d'.format(d)
    fun = context[key]
    return fun



class KroneckerProduct:

    def __init__(self, arrays):

        # right now, arrays is a list of vectors or matrices
        # TODO: allow for 3d-array
        self.arrays = [np.asarray(a) for a in arrays]
        self.d = len(arrays)
        self.tensor_type = self.arrays[0].ndim
        # 1: tensor product of vectors
        # 2: vectorized tensor product
        # 3: vectorized tensor product with derivative informations

    def as_matrix(self):
        if self.tensor_type < 3:
            return gu_tensor_product(self.arrays)
        else:
            raise Exception('Not supported. Use as_array instead.')

    def as_array(self, enum='complete'):
        if enum == 'complete':
            # TODO: check that all arrays have the same first and last dimensions
            rr = np.arange(self.arrays[0].shape[2])
            enum = [tuple(el.astype(int).tolist()) for el in cartesian([rr]*self.d, order='F') if sum(el)<=1]
        last_dim = len(enum)
        N = self.arrays[0].shape[0]
        K = reduce(operator.mul, [a.shape[1] for a in self.arrays])
        res = np.zeros((N, K, last_dim))
        for k in range(last_dim):
            arrs = [a[:, :, 0] for a in self.arrays]
            rhs = gu_tensor_product(arrs)
            res[:, :, k] = rhs
        return res

    def __mul__(self, c):
        c = np.asarray(c)
        # placeholder algo
        if self.tensor_type <3:
            mat = self.as_matrix()
            res = mat @ c.reshape((mat.shape[1],-1))
            if c.ndim == len(self.arrays):
                res = res.ravel()
        else:
            mat = self.as_array()
            tmat = mat.swapaxes(1,2)

            res = np.dot( tmat , c.reshape((mat.shape[1],-1)) )
            if c.ndim == 1:
                res = res[:,:,0]
            elif c.ndim==2:
                res = res.swapaxes(1,2)
        return res

def gu_tensor_product(arrays):
    # this is actually a "kronecker" product
    # placeholder algo
    d = len(arrays)
    if d == 1:
        return arrays[0]
    tensor_type = arrays[0].ndim
    c = tensor_type - 1
    # enum_res = (np.expand_dims(a, axis=d-i-1+c) for i, a in enumerate(arrays))
    enum_res = []
    for i,a in enumerate(arrays):
        ind = [None]*d
        ind[i] = slice(None,None,None)
        if tensor_type == 2:
            ind = [slice(None,None,None)] + ind
        enum_res.append(a[ind])
    res = reduce(operator.mul, enum_res)
    if tensor_type ==1:
        return res.ravel()
    else:
        return  res.reshape((res.shape[0],-1))
