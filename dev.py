from interpolation.splines.codegen import get_code_spline


txt = get_code_spline(1, vectorized=False, vector_valued=True, allocate=False, orders=None) # orders=((0,),(1,)))

print(txt)

