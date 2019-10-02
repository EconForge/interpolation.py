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

def source_to_function(source, context={}):
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

###
# Template to compute basis functions
###

def blending_formula(k=1, l=0, i=0):
    """
    k: spline order
    l: diff order
    i: current dimension
    """
    if k==1:
        if l==0:
            s = f"""\
Φ_{i}_{0} = 1.0 - λ_{i}
Φ_{i}_{1} = λ_{i}"""
        elif l==1:
            s = f"""\
d_Φ_{i}_{0} = -1.0/δ_{i}
d_Φ_{i}_{1} = 1.0/δ_{i}"""
        else:
            s = f"""
d_{{l}}_Φ_{i}_{0} = 0.0
d_{{l}}_Φ_{i}_{1} = 0.0"""
    elif k==3:
        if l>1:
            raise Exception("Not implemented")
        import tempita
        template = tempita.Template("""
μ_{{i}}_0 = λ_{{i}}*λ_{{i}}*λ_{{i}};  μ_{{i}}_1 = λ_{{i}}*λ_{{i}};  μ_{{i}}_2 = λ_{{i}};  μ_{{i}}_3 = 1.0;
{{for j in range(4)}}
Φ_{{i}}_{{j}} = 0.0
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
    Φ_{{i}}_{{j}} = (Cd[{{j}},0]*μ_{{i}}_0 + Cd[{{j}},1]*μ_{{i}}_1 + Cd[{{j}},2]*μ_{{i}}_2 + Cd[{{j}},3]*μ_{{i}}_3)
    {{endfor}}
""")
        s = template.substitute(i=i)

    return s


##############
# Templates  #
##############


template_linear = """
def eval_{{'linear' if k==1 else 'cubic'}}(grid, C, points{{ ', out' if not allocate else ""}}{{ ', extrap_mode' if extrap_mode else ""}}):
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
    # dim {{i}}: uniform
    a_{{i}} = grid[{{i}}][0]
    b_{{i}} = grid[{{i}}][1]
    n_{{i}} = grid[{{i}}][2]
    δ_{{i}} = (n_{{i}}-1.0)/(b_{{i}}-a_{{i}})
    {{elif grid_types[i]=='nonuniform'}}
    # dim {{i}}: nonuniform
    gc_{{i}} = grid[{{i}}]
    N_{{i}} = gc_{{i}}.shape[0]
    {{endif}}
    {{endfor}}

    # extract coordinates
    {{for i in range(d)}}
    x_{{i}} = points[{{i}}]
    {{if extrap_mode == 'nearest'}}
    x_{{i}} = max(a_{{i}}, min(x_{{i}}, b_{{i}}))
    {{endif}}
    {{if extrap_mode == 'constant'}}
    if (x_{{i}}<a_{{i}}) or (x_{{i}}>b_{{i}}):
        {{if vector_valued}}
        out[:] = 0.0
        {{if allocate}}
        return out
        {{endif}}
        {{else}}
        val = 0.0
        return val
        {{endif}}
    {{endif}}
    {{endfor}}

    # compute indices and barycentric coordinates
    {{for i in range(d)}}

    {{if (not grid_types) or (grid_types[i]=='uniform')}}
    # dimension {{i}}: uniform grid
    u_{{i}} = (x_{{i}} - a_{{i}})*δ_{{i}}
    i_{{i}} = int( floor( u_{{i}} ) )
    i_{{i}} = max( min(i_{{i}},n_{{i}}-2), 0 )
    λ_{{i}} = u_{{i}}-i_{{i}}
    {{elif grid_types[i]=='nonuniform'}}
    # dimension {{i}}: nonuniform grid
    i_{{i}} = int(np.searchsorted(gc_{{i}}, x_{{i}}))-1
    i_{{i}} = min(max(i_{{i}}, 0), N_{{i}}-2)
    δ_{{i}} = (gc_{{i}}[i_{{i}}+1]-gc_{{i}}[i_{{i}}])
    λ_{{i}} = (x_{{i}}-gc_{{i}}[i_{{i}}])/δ_{{i}}
    {{endif}}
    {{endfor}}

    # Basis functions
    {{for i in range(d)}}
    {{for l in bases_orders[i]}}
{{indent(blending_formula(k,l,i=i),levels=1)}}
    {{endfor}}
    {{endfor}}


    {{if vector_valued}}

    # compute tensor reductions
    for i_x in range(n_vals):
        out[i_x] = {{ get_values(d, multispline=True, k=k+1, i_x='i_x') }}

    {{if allocate}}
    return out
    {{endif}}

    {{else}}

    val = {{get_values(d, multispline=False, k=k+1)}}
    return val

    {{endif}}
"""

template_linear_vectorized = """
def eval_{{'linear' if k==1 else 'cubic'}}(grid, C, points{{', out' if not allocate else ""}}{{ ', extrap_mode' if extrap_mode else ""}}):
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
    # dim {{i}}: uniform
    a_{{i}} = grid[{{i}}][0]
    b_{{i}} = grid[{{i}}][1]
    n_{{i}} = grid[{{i}}][2]
    δ_{{i}} = (n_{{i}}-1.0)/(b_{{i}}-a_{{i}})
    {{elif grid_types[i]=='nonuniform'}}
    # dim {{i}}: nonuniform
    gc_{{i}} = grid[{{i}}]
    N_{{i}} = gc_{{i}}.shape[0]
    a_{{i}} = gc_{{i}}[0]
    b_{{i}} = gc_{{i}}[N_{{i}}-1]
    {{endif}}
    {{endfor}}

    for nn in range(N):

        # extract coordinates
        {{for i in range(d)}}
        x_{{i}} = points[nn,{{i}}]
        {{if extrap_mode == 'nearest'}}
        x_{{i}} = max(a_{{i}}, min(x_{{i}}, b_{{i}}))
        {{endif}}
        {{if extrap_mode == 'constant'}}
        if (x_{{i}}<a_{{i}}) or (x_{{i}}>b_{{i}}):
            {{if vector_valued}}
            out[nn,:] = 0.0
            {{else}}
            out[nn] = 0.0
            {{endif}}
            continue
        {{endif}}
        {{endfor}}

        # compute indices and barycentric coordinates
        {{for i in range(d)}}

        {{if (not grid_types) or (grid_types[i]=='uniform')}}
        # dimension {{i}}: uniform grid
        u_{{i}} = (x_{{i}} - a_{{i}})*δ_{{i}}
        i_{{i}} = int( floor( u_{{i}} ) )
        i_{{i}} = max( min(i_{{i}},n_{{i}}-2), 0 )
        λ_{{i}} = u_{{i}}-i_{{i}}
        {{elif grid_types[i]=='nonuniform'}}
        # dimension {{i}}: nonuniform grid
        i_{{i}} = int(np.searchsorted(gc_{{i}}, x_{{i}}))-1
        i_{{i}} = min(max(i_{{i}}, 0), N_{{i}}-2)
        δ_{{i}} = (gc_{{i}}[i_{{i}}+1]-gc_{{i}}[i_{{i}}])
        λ_{{i}} = (x_{{i}}-gc_{{i}}[i_{{i}}])/δ_{{i}}
        {{endif}}
        {{endfor}}

        # Basis functions
        {{for i in range(d)}}
        {{for l in bases_orders[i]}}
{{indent(blending_formula(k,l,i=i),levels=2)}}
        {{endfor}}
        {{endfor}}

        {{if vector_valued}}
        for i_x in range(n_vals):
            out[nn,i_x] = {{get_values(d, multispline=True, k=k+1)}}
        {{else}}

        out[nn] = {{get_values(d, multispline=False, k=k+1)}}
        {{endif}}


    {{if allocate}}
    return out
    {{endif}}
"""

import tempita



def get_code_spline(d, k=1, vector_valued=False, vectorized=False, allocate=False, grid_types=None, extrap_mode=None, orders=None):
    if orders is None:
        orders = (0,)*d
    # bases function to compute for each dimension
    if isinstance(orders, tuple):
        oo = [orders]
    else:
        oo = orders
    bases_orders = [sorted(list(set(e))) for e in zip(*oo)]

    if grid_types is None:
        grid_types = ['uniform']*d

    if set(grid_types) != set(['uniform']) and k>1:
        raise Exception("Nonuniform grids are only supported for linear splines.")



    templ = tempita.Template(template_linear)
    templ_vec = tempita.Template(template_linear_vectorized)
    if vectorized:
        template = templ_vec
    else:
        template = templ

    code = ( template.substitute(d=d, vector_valued=vector_valued, get_values=get_values,
            allocate=allocate, grid_types=grid_types, extrap_mode=extrap_mode, orders=orders, bases_orders=bases_orders,
        blending_formula=blending_formula, indent=indent, k=k) )
    return (code)[1:]

def get_code_linear(d, **kwargs):
    kw = {}
    kw.update(**kwargs)

    return get_code_spline(d, **kw)

def get_code_cubic(d, **kwargs):
    kw = {}
    kw.update(**kwargs)

    return get_code_spline(d, k=3, **kw)


def indent(txt,levels=1):
    return str.join('\n', ['    '*levels + e for e in str.split( txt,"\n")])
