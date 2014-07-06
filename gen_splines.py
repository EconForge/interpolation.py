import argparse

parser = argparse.ArgumentParser(description='Process spline templates.')
parser.add_argument('-o','--max_order', type=int, default=4)
parser.add_argument('-m','--multispline', type=bool, default=True)

parser.add_argument('template',type=str, nargs='?')
parser.add_argument('output',type=str, nargs='?')

args = parser.parse_args()


ftemplate = args.template
fout = args.output

multispline = (args.multispline)
max_order = args.max_order

def print_expr(symbs, inds=[]):
    if len(symbs) == 0:
        if multispline:
            return 'coefs[k,{}]'.format(str.join(',',['i{}+{}'.format(i,k) for i,k in enumerate(inds)]))
        else:
            return 'coefs[{}]'.format(str.join(',',['i{}+{}'.format(i,k) for i,k in enumerate(inds)]))
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

with file(ftemplate) as f:
    txt = f.read()

s = tempita.sub(txt,values=values,dvalues=dvalues,max_order=max_order)

with file(fout,'w') as f:
    f.write(s)
#print(s)
