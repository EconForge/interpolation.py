max_order = 4

def print_expr(symbs, inds=[]):
    if len(symbs) == 0:
        return 'C[{}]'.format(str.join(',',['i{}+{}'.format(i,k) for i,k in enumerate(inds)]))
    else:
        h = symbs[0]
        q = symbs[1:]
        exprs = [  '{}_{}*({})'.format(h,i,print_expr(q,inds + [i])) for i in range(4)]
        return str.join( ' + ', exprs )
                
values = []
dvalues = []
for order in range(max_order+1):
    expr = print_expr( ['Phi{}'.format(i) for i in range(order)] )
    values.append( expr )
    dv = []
    for i in range(order):
        args =  ['Phi{}'.format(h) for h in range(order)]
        args[i] = 'dPhi{}'.format(i)
        dexpr = print_expr( args )
        dv.append(dexpr)
    dvalues.append(dv)

import tempita

with file('splines_cython.pyx.in') as f:
    txt = f.read()

s = tempita.sub(txt,values=values,dvalues=dvalues,max_order=max_order)

with file('splines_cython.pyx','w') as f:
    f.write(s)
#print(s)


