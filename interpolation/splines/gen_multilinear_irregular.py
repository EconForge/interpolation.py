max_d = 5
# use_open_mp = True

def index(inds):
    return str.join('',  [str(e) for e in inds] )

def rindex(binds):
    # M = ['M_{}*'.format(i) for i in range(len(binds)-1)] + ['']

    N = ['(q_{}{})'.format(n,'+1'*i) for n,i in enumerate(binds)]
    # return str.join(' , ',  [ str.join('', e) for e in zip(M,N) ])
    return str.join(' , ',  N )

def make_formula(d,ind,mm):
    if len(ind) == d:
        return 'v_{}'.format(index(ind))
    else:
        j = len(ind)
        ind1 = ind + (0,)
        ind2 = ind + (1,)
        s = "(1-lam_{j})*({a}) + (lam_{j})*({b})".format(j=j, a=make_formula(d,ind1,mm), b=make_formula(d,ind2,mm))
        return s

formulas = [make_formula(i,tuple([]),None) for i in range(max_d+1)]
from itertools import product


with open('multilinear_irregular_numba.py.in') as f:
        txt = f.read()

import tempita
# template = tempita.Template(txt,name='multilinear_numba.py.in' )
# code = template.substitute()

code = tempita.sub(txt, max_d=max_d, product=product, index=index, rindex=rindex, formulas=formulas)

with open('multilinear_irregular_numba.py','w') as f:
        f.write(code)
