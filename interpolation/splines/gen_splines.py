import argparse

parser = argparse.ArgumentParser(description='Process spline templates.')
parser.add_argument('-o','--max_order', type=int, default=4)
parser.add_argument('-m','--multispline', type=bool, default=False)

parser.add_argument('template',type=str, nargs='?')
parser.add_argument('output',type=str, nargs='?')

args = parser.parse_args()

ftemplate = args.template
fout = args.output

multispline = (args.multispline)
max_order = args.max_order


def print_expr(symbs, inds=[], multispline=False):
    if len(symbs) == 0:
        if multispline:
            return 'coefs[{},k]'.format(str.join(',',['i{}+{}'.format(i,k) for i,k in enumerate(inds)]))
        else:
            return 'coefs[{}]'.format(str.join(',',['i{}+{}'.format(i,k) for i,k in enumerate(inds)]))
    else:
        h = symbs[0]
        q = symbs[1:]
        exprs = [  '{}_{}*({})'.format(h,i,print_expr(q,inds + [i], multispline=multispline)) for i in range(4)]
        return str.join( ' + ', exprs )

    # values = []
    # dvalues = []
    # for order in range(max_order+1):
    #     expr = print_expr( ['Phi{}'.format(i) for i in range(order)] )
    #     values.append( expr )
    #     dv = []
    #     for i in range(order):
    #         args =  ['Phi{}'.format(h) for h in range(order)]
    #         args[i] = 'dPhi{}'.format(i)
    #         dexpr = print_expr( args )
    #         dv.append(dexpr)
    #     dvalues.append(dv)


def get_values(order, multispline=False):
        values = []
        expr = print_expr( ['Phi{}'.format(i) for i in range(order)], multispline=multispline )
        return expr

def get_dvalues(order, i, multispline=False):
    args =  ['Phi{}'.format(h) for h in range(order)]
    args[i] = 'dPhi{}'.format(i)
    dexpr = print_expr( args, multispline=multispline )
    return dexpr

import tempita

with open(ftemplate) as f:
    txt = f.read()

s = tempita.sub(txt,values=get_values,dvalues=get_dvalues,max_order=max_order)

with open(fout,'w') as f:
    f.write(s)
#print(s)
