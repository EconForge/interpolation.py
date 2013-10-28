
with file('multilinear_cython.pyx.in') as f:
        txt = f.read()

import tempita
template = tempita.Template(txt,name='multilinear_cython.pyx.in' )
code = template.substitute()

with file('multilinear_cython.pyx','w') as f:
        f.write(code)
