import numpy as np
from numba import jit


# TODO: Numba bug somewhere. It tells me i is not defined???
# @jit
def lookup(table, x, p=0):
    n = table.size
    m = x.size
    out = np.empty(m, dtype=int)

    # lower enbound adjustment
    numfirst = 1
    while numfirst < n and table[numfirst] == table[0]:
        numfirst += 1

    # Upper endpoint adjustment
    if p >= 2:
        n -= 1
        for i in range(n-1, 0, -1):
            if table[i] == table[-1]:
                n -= 1
            else:
                break

    n1 = n - 1
    n2 = n - 2

    # handle 1-value lists separately
    if n - numfirst < 1:
        if p == 1 or p == 3:
            for i in range(m):
                out[i] = numfirst-1
        else:
            for i in range(m):
                if table[0] <= x[i]:
                    out[i] = numfirst-1
                else:
                    out[i] = -1

        return out

    jlo = 0

    for i in range(m):
        inc = 1
        xi = x[i]
        if xi >= table[jlo]:
            jhi = jlo + 1
            while xi >= table[jhi]:
                jlo = jhi
                jhi += inc
                if jhi >= n:
                    break
                else:
                    inc += inc
        else:
            jhi = jlo
            jlo -= 1
            while xi < table[jlo]:
                jhi = jlo
                jlo -= inc
                if jlo < 0:
                    jlo = -1
                    break
                else:
                    inc += inc

        while jhi - jlo > 1:
            j = (jhi + jlo) // 2
            # if j >= n:
            #     j = n-1
            if j < n and xi >= table[j]:
                jlo = j
            else:
                jhi = j

        out[i] = jlo

        if jlo < 0:
            jlo = 0
            if p == 1 or p == 3:
                out[i] = numfirst - 1

        if jlo == n1:
            jlo = n2

    return out


if __name__ == '__main__':
    table1 = np.array([1.0, 4.0])
    table2 = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
                       4.0])

    x = np.array([0.5, 1.0, 1.5, 4.0, 5.5])
    x2 = np.array([0.5, 2.0])

    assert all(lookup(table1, x, 0) == np.array([-1, 0, 0, 1, 1]))
    assert all(lookup(table1, x, 1) == np.array([0, 0, 0, 1, 1]))
    assert all(lookup(table1, x, 2) == np.array([-1, 0, 0, 0, 0]))
    assert all(lookup(table1, x, 3) == np.array([0, 0, 0, 0, 0]))

    assert all(lookup(table2, x, 0) == np.array([-1, 2, 2, 11, 11]))
    assert all(lookup(table2, x, 1) == np.array([2, 2, 2, 11, 11]))
    assert all(lookup(table2, x, 2) == np.array([-1, 2, 2, 7, 7]))
    assert all(lookup(table2, x, 3) == np.array([2, 2, 2, 7, 7]))

    assert all(lookup(np.array([1.0]), x2, 0) == np.array([-1, 0]))
    assert all(lookup(np.array([1.0]), x2, 1) == np.array([0, 0]))
    assert all(lookup(np.array([1.0]), x2, 2) == np.array([-1, 0]))
    assert all(lookup(np.array([1.0]), x2, 3) == np.array([0, 0]))
