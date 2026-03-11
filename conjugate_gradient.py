#!/usr/bin/env python3
"""Conjugate gradient method — iterative solver for symmetric positive definite systems."""
import sys
def dot(a, b): return sum(ai*bi for ai, bi in zip(a, b))
def matvec(A, x): return [sum(A[i][j]*x[j] for j in range(len(x))) for i in range(len(A))]
def vsub(a, b): return [ai-bi for ai, bi in zip(a, b)]
def vadd(a, b): return [ai+bi for ai, bi in zip(a, b)]
def smul(s, v): return [s*vi for vi in v]

def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=100):
    n = len(b); x = x0 or [0.0]*n
    r = vsub(b, matvec(A, x)); p = r[:]; rs_old = dot(r, r)
    for i in range(max_iter):
        Ap = matvec(A, p); alpha = rs_old / dot(p, Ap)
        x = vadd(x, smul(alpha, p)); r = vsub(r, smul(alpha, Ap))
        rs_new = dot(r, r)
        if rs_new < tol: return x, i+1
        p = vadd(r, smul(rs_new/rs_old, p)); rs_old = rs_new
    return x, max_iter

if __name__ == "__main__":
    A = [[4,1],[1,3]]; b = [1,2]
    x, iters = conjugate_gradient(A, b)
    print(f"Solve Ax = b")
    print(f"A = {A}, b = {b}")
    print(f"x = {[round(xi, 6) for xi in x]} ({iters} iterations)")
    Ax = matvec(A, x)
    print(f"Ax = {[round(xi, 6) for xi in Ax]}")
