from typing import Callable

def gen_trex(l, c: Callable, inds=[]):

    """Generate expression for tensor reduction"""
    # l = [
    #     ('a_0', 'a_1'),
    #     ('b_0', 'b_1', 'b_2')
    # ]
    # c = lambda i,j: 'C_{}_{}'.format(i,j)
    # gen_tr(l, c)

    if len(l)==0:
        return c(*inds)
    else:
        h = l[0]
        q = l[1:]
        k = len(h)
        exprs = [  '{}*({})'.format(h[i],gen_trex(q,c, inds=inds + [i])) for i in range(k)]
        return str.join( ' + ', exprs )

def get_values(d, multispline=False, k=4, i_x='i_x'):
        values = []
        l = [tuple(['Φ_{}_{}'.format(i, ik) for ik in range(k)]) for i in range(d)]
        if multispline:
            def c(*arg):
                inds = [(f'i_{i} + {j}' if j>0 else f'i_{i}') for i,j in enumerate(arg)]
                return f'C[{str.join(", ", [str(e) for e in inds])}, {i_x}]'
        else:
            def c(*arg):
                inds = [(f'i_{i} + {j}' if j>0 else f'i_{i}') for i,j in enumerate(arg)]
                return f'C[{str.join(", ", [str(e) for e in inds])}]'
        s = gen_trex(l, c)
        return s

def get_dvalues(d, diff, multispline=False, k=4, i_x='i_x'):
        values = []
        l = []
        for i in range(d):
            if i==diff:
                t = tuple(['dΦ_{}_{}'.format(i, ik) for ik in range(k)])
            else:
                t = tuple(['Φ_{}_{}'.format(i, ik) for ik in range(k)])
            l.append(t)
        if multispline:
            def c(*arg):
                inds = [(f'i_{i} + {j}' if j>0 else f'i_{i}') for i,j in enumerate(arg)]
                return f'C[{str.join(", ", [str(e) for e in inds])}, {i_x}]'
        else:
            def c(*arg):
                inds = [(f'i_{i} + {j}' if j>0 else f'i_{i}') for i,j in enumerate(arg)]
                return f'C[{str.join(", ", [str(e) for e in inds])}]'
        s = gen_trex(l, c)
        return s





txt_vec = """
def vec_eval_cubic(a, b, n, C, points, out):
    "This is my docstring"

    N = points.shape[0]

    {{if vector_valued}}
    n_vals = C.shape[{{d}}]
    {{endif}}

    #recover grid parameters
    {{for i in range(d)}}

    a_{{i}} = a[{{i}}]
    b_{{i}} = b[{{i}}]
    n_{{i}} = n[{{i}}]
    δ_{{i}} = (n_{{i}}-1.0)/(b_{{i}}-a_{{i}})
    {{endfor}}

    for n in range(N):

        # extract coordinates
        {{for i in range(d)}}
        x_{{i}} = points[n,{{i}}]
        {{endfor}}

        # compute indices and barycentric coordinates
        {{for i in range(d)}}

        u_{{i}} = (x_{{i}} - a_{{i}})*δ_{{i}}
        i_{{i}} = int( floor( u_{{i}} ) )
        i_{{i}} = max( min(i_{{i}},n_{{i}}-2), 0 )
        λ_{{i}} = u_{{i}}-i_{{i}}
        {{endfor}}


        # compute coefficients for blending formulas
        {{for i in range(d)}}
        tp_{{i}}_0 = λ_{{i}}*λ_{{i}}*λ_{{i}};  tp_{{i}}_1 = λ_{{i}}*λ_{{i}};  tp_{{i}}_2 = λ_{{i}};  tp_{{i}}_3 = 1.0;
        {{endfor}}


        {{for i in range(d)}}

        Φ_{{i}}_0 = 0
        Φ_{{i}}_1 = 0
        Φ_{{i}}_2 = 0
        Φ_{{i}}_3 = 0
        if λ_{{i}} < 0:
            Φ_{{i}}_0 = dCd[0,3]*λ_{{i}} + Cd[0,3]
            Φ_{{i}}_1 = dCd[1,3]*λ_{{i}} + Cd[1,3]
            Φ_{{i}}_2 = dCd[2,3]*λ_{{i}} + Cd[2,3]
            Φ_{{i}}_3 = dCd[3,3]*λ_{{i}} + Cd[3,3]
        elif λ_{{i}} > 1:
            Φ_{{i}}_0 = (3*Cd[0,0] + 2*Cd[0,1] + Cd[0,2])*(λ_{{i}}-1) + (Cd[0,0]+Cd[0,1]+Cd[0,2]+Cd[0,3])
            Φ_{{i}}_1 = (3*Cd[1,0] + 2*Cd[1,1] + Cd[1,2])*(λ_{{i}}-1) + (Cd[1,0]+Cd[1,1]+Cd[1,2]+Cd[1,3])
            Φ_{{i}}_2 = (3*Cd[2,0] + 2*Cd[2,1] + Cd[2,2])*(λ_{{i}}-1) + (Cd[2,0]+Cd[2,1]+Cd[2,2]+Cd[2,3])
            Φ_{{i}}_3 = (3*Cd[3,0] + 2*Cd[3,1] + Cd[3,2])*(λ_{{i}}-1) + (Cd[3,0]+Cd[3,1]+Cd[3,2]+Cd[3,3])
        else:
            Φ_{{i}}_0 = (Cd[0,0]*tp_{{i}}_0 + Cd[0,1]*tp_{{i}}_1 + Cd[0,2]*tp_{{i}}_2 + Cd[0,3]*tp_{{i}}_3)
            Φ_{{i}}_1 = (Cd[1,0]*tp_{{i}}_0 + Cd[1,1]*tp_{{i}}_1 + Cd[1,2]*tp_{{i}}_2 + Cd[1,3]*tp_{{i}}_3)
            Φ_{{i}}_2 = (Cd[2,0]*tp_{{i}}_0 + Cd[2,1]*tp_{{i}}_1 + Cd[2,2]*tp_{{i}}_2 + Cd[2,3]*tp_{{i}}_3)
            Φ_{{i}}_3 = (Cd[3,0]*tp_{{i}}_0 + Cd[3,1]*tp_{{i}}_1 + Cd[3,2]*tp_{{i}}_2 + Cd[3,3]*tp_{{i}}_3)
        {{endfor}}

        {{if vector_valued}}

        for i_x in range(n_vals):
            out[n,i_x] = {{get_values(d, multispline=True, k=4)}}
        {{else}}

        out[n] = {{get_values(d, multispline=False, k=4)}}
        {{endif}}
"""



txt = """
def eval_cubic(a, b, n, C, point, val_n):
    "This is my docstring"

    # {{if vector_valued}}
    # n_vals = C.shape[{{d}}]
    # {{endif}}

    #recover grid parameters
    {{for i in range(d)}}

    a_{{i}} = a[{{i}}]
    b_{{i}} = b[{{i}}]
    n_{{i}} = n[{{i}}]
    δ_{{i}} = (n_{{i}}-1.0)/(b_{{i}}-a_{{i}})
    {{endfor}}

    # extract coordinates
    {{for i in range(d)}}
    x_{{i}} = point[{{i}}]
    {{endfor}}

    # compute indices and barycentric coordinates
    {{for i in range(d)}}

    u_{{i}} = (x_{{i}} - a_{{i}})*δ_{{i}}
    i_{{i}} = int( floor( u_{{i}} ) )
    i_{{i}} = max( min(i_{{i}},n_{{i}}-2), 0 )
    λ_{{i}} = u_{{i}}-i_{{i}}
    {{endfor}}


    # compute coefficients for blending formulas
    {{for i in range(d)}}
    tp_{{i}}_0 = λ_{{i}}*λ_{{i}}*λ_{{i}};  tp_{{i}}_1 = λ_{{i}}*λ_{{i}};  tp_{{i}}_2 = λ_{{i}};  tp_{{i}}_3 = 1.0;
    {{endfor}}


    {{for i in range(d)}}

    Φ_{{i}}_0 = 0
    Φ_{{i}}_1 = 0
    Φ_{{i}}_2 = 0
    Φ_{{i}}_3 = 0
    if λ_{{i}} < 0:
        Φ_{{i}}_0 = dCd[0,3]*λ_{{i}} + Cd[0,3]
        Φ_{{i}}_1 = dCd[1,3]*λ_{{i}} + Cd[1,3]
        Φ_{{i}}_2 = dCd[2,3]*λ_{{i}} + Cd[2,3]
        Φ_{{i}}_3 = dCd[3,3]*λ_{{i}} + Cd[3,3]
    elif λ_{{i}} > 1:
        Φ_{{i}}_0 = (3*Cd[0,0] + 2*Cd[0,1] + Cd[0,2])*(λ_{{i}}-1) + (Cd[0,0]+Cd[0,1]+Cd[0,2]+Cd[0,3])
        Φ_{{i}}_1 = (3*Cd[1,0] + 2*Cd[1,1] + Cd[1,2])*(λ_{{i}}-1) + (Cd[1,0]+Cd[1,1]+Cd[1,2]+Cd[1,3])
        Φ_{{i}}_2 = (3*Cd[2,0] + 2*Cd[2,1] + Cd[2,2])*(λ_{{i}}-1) + (Cd[2,0]+Cd[2,1]+Cd[2,2]+Cd[2,3])
        Φ_{{i}}_3 = (3*Cd[3,0] + 2*Cd[3,1] + Cd[3,2])*(λ_{{i}}-1) + (Cd[3,0]+Cd[3,1]+Cd[3,2]+Cd[3,3])
    else:
        Φ_{{i}}_0 = (Cd[0,0]*tp_{{i}}_0 + Cd[0,1]*tp_{{i}}_1 + Cd[0,2]*tp_{{i}}_2 + Cd[0,3]*tp_{{i}}_3)
        Φ_{{i}}_1 = (Cd[1,0]*tp_{{i}}_0 + Cd[1,1]*tp_{{i}}_1 + Cd[1,2]*tp_{{i}}_2 + Cd[1,3]*tp_{{i}}_3)
        Φ_{{i}}_2 = (Cd[2,0]*tp_{{i}}_0 + Cd[2,1]*tp_{{i}}_1 + Cd[2,2]*tp_{{i}}_2 + Cd[2,3]*tp_{{i}}_3)
        Φ_{{i}}_3 = (Cd[3,0]*tp_{{i}}_0 + Cd[3,1]*tp_{{i}}_1 + Cd[3,2]*tp_{{i}}_2 + Cd[3,3]*tp_{{i}}_3)
    {{endfor}}

    {{if vector_valued}}

    # compute tensor reductions
        {{for i_x in range(n_vals)}}
    val_{{i_x}} = {{ get_values(d, multispline=True, k=4, i_x=i_x) }}
        {{endfor}}

    val = (
        {{for i_x in range(n_vals)}}
        val_{{i_x}},
        {{endfor}}
    )

    return val

    {{else}}

    val = {{get_values(d, multispline=False, k=4)}}
    {{endif}}
"""



import tempita

templ = tempita.Template(txt)
templ_vec = tempita.Template(txt_vec)

code
#
from numba import jit, generated_jit
from numpy import zeros

Ad = np.array([
#      t^3       t^2        t        1
   [-1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0],
   [ 3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0],
   [-3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0],
   [ 1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0]
])
dAd = zeros((4,4))
for i in range(1,4):
    dAd[:,i] = Ad[:,i-1]*(4-i)

@generated_jit
def vec_eval_cubic(a,b,n,C,points,out):
    d = len(a.types)
    vector_valued = (C.ndim==d+1)
    print(vector_valued)
    context = {'floor': floor,'Cd': Ad, 'dCd': dAd}
    code = ( templ_vec.substitute(d=d, vector_valued=vector_valued, get_values=get_values) )
    print(code)
    f = source_to_function(code, context=context)
    return f

# Val = tuple([(CVal(),tuple(range(i))) for i in range(max_Val)])
import numba
max_Val = 100
Val = tuple([tuple(range(i)) for i in range(max_Val)])

tt = numba.typeof(Val[10])




%time vec_eval_cubic(a,b,n,c, points, out)

a
b

outs = np.zeros(N)
cc = c[...,0]
cc.ndim
%time vec_eval_cubic(a,b,n,cc, points, outs)

c.shape


@generated_jit
def eval_cubic(a,b,n,C,point,val_d):
    d = len(a.types)
    n_x = len(val_d.types)
    vector_valued = (C.ndim==d+1)
    context = {'floor': floor,'Cd': Ad, 'dCd': dAd}
    code = ( templ.substitute(d=d, vector_valued=vector_valued, get_values=get_values, n_vals=n_x) )
    print(code)
    f = source_to_function(code, context=context)
    return f

%time eval_cubic(a,b,n,c, points[0,:], Val[c.shape[-1]])

def source_to_function(source, context=None):
    # if context is None:
    #     d = {}
    # else:
    #     d = context
    import ast
    from math import floor
    tree = ast.parse(source)
    fundef = tree.body[0]
    funname = fundef.name
    code = compile(tree,'<string>','exec')
    exec(code, context, context)
    return context[funname]

import numpy as np

from math import floor
context = {'floor': floor,'Cd': Ad, 'dCd': dAd}

f = source_to_function(code, context=context)


d = 2
n_x= 3
a = (0.0, 0.0)
b = (1.0, 1.0)
n = (10,10)
c = np.random.random((12,12,n_x))
N = 100
points = np.random.random((N,2))
out = np.empty((N,n_x))

out1 = out.copy()
%timeit f(a,b,n,c,points,out)

from numba import njit

ff = njit(f)

out2 = out.copy()
ff(a,b,n,c,points,out)
%timeit ff(a,b,n,c,points,out)
