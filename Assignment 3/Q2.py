import numpy as np
import matplotlib.pyplot as plt

from math import ceil
from numba import njit
from scipy.sparse.linalg import LinearOperator, cg
from scipy.sparse import coo_matrix

M = 1

def isbc(s, i, j, M):
    if s == 0:
        if i == 0 or j == M - 1 or j == 0:
            return True
    if s == 1:
        if j == 0 or j == 2*M - 1:
            return True
        if i == M and j >= M - 1:
            return True
        if i == 2*M-1:
            return True
    return False


def iscorner(s, i, j, M):
    if s == 0:
        if i == 0 and (j == 0 or j == M - 1):
            return True
        return False
    if s == 1:
        if i == 2*M-1 and (j == 0 or j == 2*M-1):
            return True
        if i == M + 1 and j == 2*M-1:
            return True
        return False


def build_system(M, grad, f):
    
    data = []
    rows = []
    cols = []
    
    Mop = 3*(M-2)**2 + 4*(M-2)
    
    b = np.ones(Mop)*grad
    
    def add(val, row, colshift):
        data.append(val)
        rows.append(row)
        if row+colshift < 0:
            raise Exception(f'Negative col index {row}: {colshift}')
        cols.append(row+colshift)
        
    k = 0
    
    s = 0
    for i in range(M+1):
        for j in range(M):
            if isbc(s, i, j, M):
                if not iscorner(s, i, j, M):
                    print(k, i, j, f(i, j))
                    b[k] += f(i, j)
            else:
                # center
                add(4, k, 0)
                
                # left
                if j != 1:
                    add(-1, k, -1)
                    
                # right
                if j != M - 2:
                    add(-1, k, 1)
                
                # bottom
                if i != 1:
                    add(-1, k, -(M-2))
                
                # top
                add(-1, k, M-2)
                k += 1
    s = 1
    for i in range(M+1, 2*M):
        for j in range(M*2):
            # print('#', k, '#')
            if isbc(s, i, j, M):
                if not iscorner(s, i, j, M):
                    pass
            else:
                # check for boundaries
                # top
                if isbc(s, i+1, j, M):
                    b[k] += f(i+1, j)

                # bottom
                if isbc(s, i-1, j, M):
                    b[k] += f(i-1, j)

                # left
                if isbc(s, i, j-1, M):
                    b[k] += f(i, j-1)

                # right
                if isbc(s, i, j+1, M):
                    b[k] += f(i, j+1)
                # center
                add(4, k, 0)

                # left
                if j != 1:
                    add(-1, k, -1)

                # right
                if j != 2*M - 2:
                    add(-1, k, 1)

                # top
                if i != 2*M - 2:
                    add(-1, k, -2*(M-2))

                # bottom
                if i == M+1 and j >= 1 and j <= M-2:
                    add(-1, k, -(M-2))
                elif i != M+1:
                    add(-1, k, -(2*M-2))
                k += 1
            
    if any([x<0 for x in cols]):
        print(cols)
        raise Exception('Negative column index')

    A = coo_matrix((data, (rows, cols))).tocsr()
    
    if A.shape[0] != A.shape[1]:
        raise Exception(f'Matrix is not square: {A.shape}')
    
    if A.shape[0] != Mop:
        raise Exception(f'Matrix wrong size:{A.shape[0]}')
    
    return A, b


def solve(A, b):

    history = []
    
    def error_calc(u):
        error = np.linalg.norm(A@u - b)
        history.append(error)
        return

    sol, info = cg(A, b, callback=error_calc)
    if info > 0:
        print('Did not converge! iter:', info)
    if info < 0:
        print('There was an error in cg')
    
    return sol, history

def show(u, M):
    pass


M = 5

def f(i, j):
    if j == 0:
        return 1
    else:
        return 0

A, b = build_system(5, 0, f)
sol, history = solve(A, b)
