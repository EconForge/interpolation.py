import tempita

import numpy as np
from numpy import floor
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

#####################
# Templates (cubic) #
#####################


template_cubic = """
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


template_cubic_vectorized = """
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



######################
# Templates (linear) #
######################


template_linear = """
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
    {{if (not grid_types) or (grid_types[i]=='uniform')}}
    a_{{i}} = grid[{{i}}][0]
    b_{{i}} = grid[{{i}}][1]
    n_{{i}} = grid[{{i}}][2]
    δ_{{i}} = (n_{{i}}-1.0)/(b_{{i}}-a_{{i}})
    {{elif grid_types[i]=='nonuniform'}}
    gc_{{i}} = grid[{{i}}]
    N_{{i}} = gc_{{i}}.shape[0]
    {{endif}}
    {{endfor}}

    # extract coordinates
    {{for i in range(d)}}
    x_{{i}} = points[{{i}}]
    {{endfor}}

    # compute indices and barycentric coordinates
    {{for i in range(d)}}

    {{if (not grid_types) or (grid_types[i]=='uniform')}}
    # uniform grid
    u_{{i}} = (x_{{i}} - a_{{i}})*δ_{{i}}
    i_{{i}} = int( floor( u_{{i}} ) )
    i_{{i}} = max( min(i_{{i}},n_{{i}}-2), 0 )
    λ_{{i}} = u_{{i}}-i_{{i}}
    {{elif grid_types[i]=='nonuniform'}}
    # nonuniform grid
    i_{{i}} = int(np.searchsorted(gc_{{i}}, x_{{i}}))-1
    i_{{i}} = min(max(i_{{i}}, 0), N_{{i}}-2)
    λ_{{i}} = (x_{{i}}-gc_{{i}}[i_{{i}}])/(gc_{{i}}[i_{{i}}+1]-gc_{{i}}[i_{{i}}])
    {{endif}}
    {{endfor}}

    # Basis functions
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

template_linear_vectorized = """
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
    {{if (not grid_types) or (grid_types[i]=='uniform')}}
    a_{{i}} = grid[{{i}}][0]
    b_{{i}} = grid[{{i}}][1]
    n_{{i}} = grid[{{i}}][2]
    δ_{{i}} = (n_{{i}}-1.0)/(b_{{i}}-a_{{i}})
    {{elif grid_types[i]=='nonuniform'}}
    gc_{{i}} = grid[{{i}}]
    N_{{i}} = gc_{{i}}.shape[0]
    {{endif}}
    {{endfor}}

    for nn in range(N):

        # extract coordinates
        {{for i in range(d)}}
        x_{{i}} = points[nn,{{i}}]
        {{endfor}}

        # compute indices and barycentric coordinates
        {{for i in range(d)}}

        {{if (not grid_types) or (grid_types[i]=='uniform')}}
        # uniform grid
        u_{{i}} = (x_{{i}} - a_{{i}})*δ_{{i}}
        i_{{i}} = int( floor( u_{{i}} ) )
        i_{{i}} = max( min(i_{{i}},n_{{i}}-2), 0 )
        λ_{{i}} = u_{{i}}-i_{{i}}
        {{elif grid_types[i]=='nonuniform'}}
        # nonuniform grid
        i_{{i}} = int(np.searchsorted(gc_{{i}}, x_{{i}}))-1
        i_{{i}} = min(max(i_{{i}}, 0), N_{{i}}-2)
        λ_{{i}} = (x_{{i}}-gc_{{i}}[i_{{i}}])/(gc_{{i}}[i_{{i}}+1]-gc_{{i}}[i_{{i}}])
        {{endif}}
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

import tempita


def get_code_cubic(d, vector_valued=False, vectorized=False, allocate=False):
    templ = tempita.Template(template_cubic)
    templ_vec = tempita.Template(template_cubic_vectorized)
    if vectorized:
        code = ( templ_vec.substitute(d=d, vector_valued=vector_valued, get_values=get_values, allocate=allocate) )
    else:
        code = ( templ.substitute(d=d, vector_valued=vector_valued, get_values=get_values, allocate=allocate) )
    return (code)[1:]



def get_code_linear(d, vector_valued=False, vectorized=False, allocate=False, grid_types=None):
    templ = tempita.Template(template_linear)
    templ_vec = tempita.Template(template_linear_vectorized)
    if vectorized:
        code = ( templ_vec.substitute(d=d, vector_valued=vector_valued, get_values=get_values, allocate=allocate, grid_types=grid_types) )
    else:
        code = ( templ.substitute(d=d, vector_valued=vector_valued, get_values=get_values, allocate=allocate, grid_types=grid_types) )
    return (code)[1:]

print(get_code_linear(2, grid_types=['uniform','nonuniform']))
# print(get_code_cubic(2))
