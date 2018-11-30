import numpy as np
from numba import njit
from typing import Callable
from math import floor

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
def eval_cubic(grid, C, points{{', out' if not allocate else ""}}):
    "This is my docstring"

    N = points.shape[0]

    {{if vector_valued}}
    n_vals = C.shape[{{d}}]
    {{endif}}

    {{if allocate}}
    {{if vector_valued}}
    out = zeros((N, n_vals))
    {{else}}
    out = zeros(N)
    {{endif}}
    {{endif}}

    #recover grid parameters
    {{for i in range(d)}}

    a_{{i}} = grid[{{i}}][0]
    b_{{i}} = grid[{{i}}][1]
    n_{{i}} = grid[{{i}}][2]
    δ_{{i}} = (n_{{i}}-1.0)/(b_{{i}}-a_{{i}})
    {{endfor}}

    for nn in range(N):

        # extract coordinates
        {{for i in range(d)}}
        x_{{i}} = points[nn,{{i}}]
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
        {{for j in range(4)}}
        Φ_{{i}}_{{j}} = 0
        {{endfor}}
        if λ_{{i}} < 0:
            {{for j in range(4)}}
            Φ_{{i}}_{{j}} = dCd[{{j}},3]*λ_{{i}} + Cd[{{j}},3]
            {{endfor}}
        elif λ_{{i}} > 1:
            {{for j in range(4)}}
            Φ_{{i}}_{{j}} = (3*Cd[{{j}},0] + 2*Cd[{{j}},1] + Cd[{{j}},2])*(λ_{{i}}-1) + (Cd[{{j}},0]+Cd[{{j}},1]+Cd[{{j}},2]+Cd[{{j}},3])
            {{endfor}}
        else:
            {{for j in range(4)}}
            Φ_{{i}}_{{j}} = (Cd[{{j}},0]*tp_{{i}}_0 + Cd[{{j}},1]*tp_{{i}}_1 + Cd[{{j}},2]*tp_{{i}}_2 + Cd[{{j}},3]*tp_{{i}}_3)
            {{endfor}}
        {{endfor}}

        {{if vector_valued}}

        for i_x in range(n_vals):
            out[nn,i_x] = {{get_values(d, multispline=True, k=4)}}
        {{else}}

        out[nn] = {{get_values(d, multispline=False, k=4)}}
        {{endif}}


    {{if allocate}}
    return out
    {{endif}}
"""



txt = """
def eval_cubic(grid, C, points{{ ', out' if not allocate else ""}}):
    "This is my docstring"

    {{if vector_valued}}
    n_vals = C.shape[{{d}}]
    {{if allocate}}
    out = zeros(n_vals)
    {{endif}}
    {{endif}}

    #recover grid parameters
    {{for i in range(d)}}

    a_{{i}} = grid[{{i}}][0]
    b_{{i}} = grid[{{i}}][1]
    n_{{i}} = grid[{{i}}][2]
    δ_{{i}} = (n_{{i}}-1.0)/(b_{{i}}-a_{{i}})
    {{endfor}}

    # extract coordinates
    {{for i in range(d)}}
    x_{{i}} = points[{{i}}]
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
    {{for j in range(4)}}
    Φ_{{i}}_{{j}} = 0
    {{endfor}}
    if λ_{{i}} < 0:
        {{for j in range(4)}}
        Φ_{{i}}_{{j}} = dCd[{{j}},3]*λ_{{i}} + Cd[{{j}},3]
        {{endfor}}
    elif λ_{{i}} > 1:
        {{for j in range(4)}}
        Φ_{{i}}_{{j}} = (3*Cd[{{j}},0] + 2*Cd[{{j}},1] + Cd[{{j}},2])*(λ_{{i}}-1) + (Cd[{{j}},0]+Cd[{{j}},1]+Cd[{{j}},2]+Cd[{{j}},3])
        {{endfor}}
    else:
        {{for j in range(4)}}
        Φ_{{i}}_{{j}} = (Cd[{{j}},0]*tp_{{i}}_0 + Cd[{{j}},1]*tp_{{i}}_1 + Cd[{{j}},2]*tp_{{i}}_2 + Cd[{{j}},3]*tp_{{i}}_3)
        {{endfor}}
    {{endfor}}

    {{if vector_valued}}

    # compute tensor reductions
    for i_x in range(n_vals):
        out[i_x] = {{ get_values(d, multispline=True, k=4, i_x='i_x') }}

    {{if allocate}}
    return out
    {{endif}}

    {{else}}

    val = {{get_values(d, multispline=False, k=4)}}
    return val

    {{endif}}
"""


txt_vec_1 = """
def eval_linear(grid, C, points{{', out' if not allocate else ""}}):
    "This is my docstring"

    N = points.shape[0]

    {{if vector_valued}}
    n_vals = C.shape[{{d}}]
    {{endif}}

    {{if allocate}}
    {{if vector_valued}}
    out = zeros((N, n_vals))
    {{else}}
    out = zeros(N)
    {{endif}}
    {{endif}}

    #recover grid parameters
    {{for i in range(d)}}

    a_{{i}} = grid[{{i}}][0]
    b_{{i}} = grid[{{i}}][1]
    n_{{i}} = grid[{{i}}][2]
    δ_{{i}} = (n_{{i}}-1.0)/(b_{{i}}-a_{{i}})
    {{endfor}}

    for nn in range(N):

        # extract coordinates
        {{for i in range(d)}}
        x_{{i}} = points[nn,{{i}}]
        {{endfor}}

        # compute indices and barycentric coordinates
        {{for i in range(d)}}
        u_{{i}} = (x_{{i}} - a_{{i}})*δ_{{i}}
        i_{{i}} = int( floor( u_{{i}} ) )
        i_{{i}} = max( min(i_{{i}},n_{{i}}-2), 0 )
        λ_{{i}} = u_{{i}}-i_{{i}}
        {{endfor}}

        {{for i in range(d)}}
        Φ_{{i}}_{{0}} = 1 - λ_{{i}}
        Φ_{{i}}_{{1}} = λ_{{i}}
        {{endfor}}

        {{if vector_valued}}
        for i_x in range(n_vals):
            out[nn,i_x] = {{get_values(d, multispline=True, k=2)}}
        {{else}}

        out[nn] = {{get_values(d, multispline=False, k=2)}}
        {{endif}}


    {{if allocate}}
    return out
    {{endif}}
"""


txt_1 = """
def eval_linear(grid, C, points{{ ', out' if not allocate else ""}}):
    "This is my docstring"

    {{if vector_valued}}
    n_vals = C.shape[{{d}}]
    {{if allocate}}
    out = zeros(n_vals)
    {{endif}}
    {{endif}}

    #recover grid parameters
    {{for i in range(d)}}

    a_{{i}} = grid[{{i}}][0]
    b_{{i}} = grid[{{i}}][1]
    n_{{i}} = grid[{{i}}][2]
    δ_{{i}} = (n_{{i}}-1.0)/(b_{{i}}-a_{{i}})
    {{endfor}}

    # extract coordinates
    {{for i in range(d)}}
    x_{{i}} = points[{{i}}]
    {{endfor}}

    # compute indices and barycentric coordinates
    {{for i in range(d)}}

    u_{{i}} = (x_{{i}} - a_{{i}})*δ_{{i}}
    i_{{i}} = int( floor( u_{{i}} ) )
    i_{{i}} = max( min(i_{{i}},n_{{i}}-2), 0 )
    λ_{{i}} = u_{{i}}-i_{{i}}
    {{endfor}}

    {{for i in range(d)}}
    Φ_{{i}}_{{0}} = 1 - λ_{{i}}
    Φ_{{i}}_{{1}} = λ_{{i}}
    {{endfor}}


    {{if vector_valued}}

    # compute tensor reductions
    for i_x in range(n_vals):
        out[i_x] = {{ get_values(d, multispline=True, k=2, i_x='i_x') }}

    {{if allocate}}
    return out
    {{endif}}

    {{else}}

    val = {{get_values(d, multispline=False, k=2)}}
    return val

    {{endif}}
"""



# %time eval_cubic(a,b,n,c, points[0,:], Val[c.shape[-1]])
from numpy import floor
import numpy

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
    tree = ast.fix_missing_locations(tree)
    code = compile(tree,'<string>','exec')
    exec(code, context, context)
    return context[funname]

import numpy as np

#
#
# #
# #
#
# # %%
# # test basic versions

#
#
# from numpy import floor
#
# code = ( templ_vec.substitute(d=2, vector_valued=0, get_values=get_values, n_vals=0) )[1:]
# print(code)
#
# f = source_to_function(code)
# f(grid, C, points ,out)


#
# # %%
#
#
# # not type-inferrable
# # eval_cubic(a,b,n,C,point) = eval_cubic(a,b,n,C,point,Val[C.shape[len(a)]])
# from numba.extending import overload
#
# def eval_cubicc():
#     pass
#
# @overload(eval_cubicc)
# def _eval_cubicc(a,b,n,C,points):
#     if points.ndim==2:
#         if C.ndim == len(a.types)+1:
#             def __eval_cubic(a,b,n,C,points):
#                 N = points.shape[0]
#                 n_x = C.shape[len(a)]
#                 out = zeros((N,n_x))
#                 vec_eval_cubic(a,b,n,C,points,out)
#                 return out
#         else:
#             def __eval_cubic(a,b,n,C,points):
#                 N = points.shape[0]
#                 out = zeros(N)
#                 vec_eval_cubic(a,b,n,C,points,out)
#                 return out
#         return __eval_cubic
#
#
# @overload(eval_cubicc)
# def _eval_cubicc(a,b,n,C,point):
#     if point.ndim==1:
#         if C.ndim == len(a.types)+1:
#             def __eval_cubic(a,b,n,C,point):
#
#                 N = points.shape[0]
#                 n_x = C.shape[len(a)]
#                 return eval_cubic(a,b,n,C,point,Val[n_x])
#             return __eval_cubic
#         else:
#             context = {'floor': floor,'Cd': Ad, 'dCd': dAd}
#             code = ( templ.substitute(d=d, vector_valued=False, get_values=get_values, n_vals=0) )
#             f = source_to_function(code, context=context)
#             return f
#
#
# @njit
# def cubic_eval(*args):
#     return eval_cubicc(*args)
#
#
# # tests
#
# # def test_eval_cubic():
#
# d = 2
# n_x= 3
# a = (0.0, 0.0)
# b = (1.0, 1.0)
# n = (10,10)
# c = np.random.random((12,12,n_x))
# cc = c[..., 0]
# N = 100
# points = np.random.random((N,2))
# out = np.empty((N,n_x))
#
#
#
#
#
# ### vectorized calls
#
# ## vector-valued
#
# # allocating
# res = cubic_eval(a,b,n,c,points)
#
# out0 = out*0
# # non-allocating
# cubic_eval(a,b,n,c,points,out0)
#
# ## single-valued
#
# # allocating
# res = cubic_eval(a,b,n,cc,points)
#
# #non-allocating
# out = res*0
# cubic_eval(a,b,n,cc,points,out)
#
#
# ### one call
#
# point = points[0,:]
#
# ## vector-valued
#
# # does not work
# res = cubic_eval(a,b,n,c,point)
# res
#
# eval_cubic(a,b,n,c,point,Val[3])
#
# res
#
# ## single-valued
# res = cubic_eval(a,b,n,cc,point)
# res
#
#
# test_eval_cubic()
#
# from math import floor
# context = {'floor': floor,'Cd': Ad, 'dCd': dAd}
#
# f = source_to_function(code, context=context)
#
#
#
# out1 = out.copy()
# %timeit f(a,b,n,c,points,out)
#
# from numba import njit
#
# ff = njit(f)
#
# out2 = out.copy()
# ff(a,b,n,c,points,out)
# %timeit ff(a,b,n,c,points,out)
