if True:

    import numpy as np
    d = 3
    K = 50
    N = 10**6
    a = np.zeros(3)
    b = np.ones(3)
    orders = np.array([K for i in range(d)])
    coeffs = np.random.random([k+2 for k in orders])
    points = np.random.random((N,d))  # each line is a vector
    points_c = points.T.copy() # each column is a vector
    vals = np.zeros(N)

    print(points.max().max())    
    print(points.min().min())    

    import time

    from alternative_implementations import *
    from eval_cubic_splines_cython import vec_eval_cubic_spline_3 as rr

    vec_eval_cubic_spline_3        (a,b,orders,coeffs,points,vals)  # warmup
    vec_eval_cubic_spline_3_inlined(a,b,orders,coeffs,points,vals)  # warmup
    vec_eval_cubic_spline_3_inlined_columns(a,b,orders,coeffs,points_c,vals)  # warmup
    vec_eval_cubic_spline_3_kernel (a,b,orders,coeffs,points,vals)  # warmup
    vec_eval_cubic_spline_3_inlined_lesswork(orders,coeffs,points,vals,Ad,dAd)
    #rr(a,b,orders,coeffs,points,vals,Ad,dAd)
    rr(a,b,orders,coeffs,points,vals)

    t1 = time.time()
    vec_eval_cubic_spline_3(a,b,orders,coeffs,points,vals)
    t2 = time.time()
    vec_eval_cubic_spline_3_inlined(a,b,orders,coeffs,points,vals)
    t3 = time.time()
    vec_eval_cubic_spline_3_inlined_columns(a,b,orders,coeffs,points_c,vals)
    t4 = time.time()
    vec_eval_cubic_spline_3_kernel(a,b,orders,coeffs,points,vals)
    t5 = time.time()
    vec_eval_cubic_spline_3_inlined_lesswork(orders,coeffs,points,vals,Ad,dAd)
    t6 = time.time()
#    rr(a,b,orders,coeffs,points,vals,Ad,dAd)
    rr(a,b,orders,coeffs,points,vals)
    t7 = time.time()
    print("one function call per point: {}".format(t2-t1))
    print("inlined (points in rows): {}".format(t3-t2))
    print("inlined (points in columns): {}".format(t4-t3))
    print("kernel: {}".format(t5-t4))
    print("less work: {}".format(t6-t5))
    print("cython: {}".format(t7-t6))
    print(vals[:10,0])


