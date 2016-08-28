import tempita

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
