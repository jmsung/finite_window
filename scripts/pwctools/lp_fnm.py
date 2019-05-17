"""
Ported by Massimo Vassalli [http://mv.nanoscopy.eu massimo.vassalli@gmail.com]
"""
# Solves a linear program by the interior-point method:
# min(c * u), s.t. A * x = b and 0 < x < u
# Based on Matlab code originally written by Daniel Morillo, Roger Koenker, and
# Paul Eilers, 1999-2004.

import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix
import numpy.linalg as linalg
from scipy.sparse.linalg import lsqr as spsolverect

def bound(x, dx):
    x = np.array(x)
    dx = np.array(dx)
    b = 1.0e20 + np.zeros(dx.shape)
    f = np.where(dx < 0)
    b[f] = -x[f] / dx[f]
    return b
        
def lp_fnm(A, c, b, u, x, stopgap=1e-5, stopit=60, display=True):

    # Set some constants
    beta = 0.9995
    m,n = A.shape

    # Generate inital feasible point
    s = u - x
    y = spsolverect(A.transpose(),c.transpose().toarray())[0].transpose()
    r = c - y * A
    idx = np.where(r==0)
    r[idx] = 0.001
    idx = np.where(r>0)
    z = np.zeros(r.shape)
    z[idx] = r[idx]
    w = z - r
    
    gap = c.dot(x) - y.dot(b) + w.dot(u)
    w = np.array(w).reshape(n)
    z = np.array(z).reshape(n)
    # Interior-point iteration loop
    it = 0
    while ((gap > stopgap) and (it < stopit)):

        it = it + 1
        
        # Compute affine step
        
        q = 1.0 / (z.reshape(n) / x.reshape(n) + w.reshape(n) / s.reshape(n))
        r = z - w
        Q = spdiags(np.sqrt(q), 0, n, n)
        AQ = A * Q
        rhs = Q * r.transpose()
        dy = spsolverect(AQ.transpose(),np.array(rhs) )[0].transpose()
        dx = (np.array(q).reshape(n) * np.array(dy * A - r).reshape(n)).transpose()
        ds = -dx
        dz = -np.array(z).reshape(n) * np.array(1.0 + dx / x).reshape(n)
        dw = -np.array(w).reshape(n) * np.array(1.0 + ds / s).reshape(n)
        
        # Compute maximum allowable step lengths
        fx = bound(x, dx);
        fs = bound(s, ds);
        fw = bound(w, dw);
        fz = bound(z, dz);
        fp = np.min([fx, fs]);
        fd = np.min([fw, fz]);
        fp = np.min([np.min(beta * fp), 1.0])
        fd = np.min([np.min(beta * fd), 1.0])
        
        # If full step is feasible, take it. Otherwise modify it
        if (np.min([fp, fd]) < 1.0):
            
            # Update mu
            mu = z * x + w * s
            g = (z + fd * dz) * (x + fp * dx) + (w + fd * dw) * (s + fp * ds)
            mu = mu * (g / mu) **3 / ( 2.0* n)
            
            # Compute modified step
            dxdz = dx * dz.transpose()
            dsdw = ds * dw.transpose()
            xinv = 1.0 / x
            sinv = 1.0 / s
            xi = mu * (xinv - sinv)
            rhs = rhs + Q * (dxdz - dsdw - xi)
            dy = spsolverect(AQ.transpose(),np.array(rhs) )[0].transpose()
            dx = q * (A.transpose() * dy.transpose() + xi - r.transpose() -dxdz + dsdw)
            ds = -dx
            dz = mu * xinv.transpose() - z - xinv.transpose() * z * dx.transpose() - dxdz.transpose()
            dw = mu * sinv.transpose() - w - sinv.transpose() * w * ds.transpose() - dsdw.transpose()
            
            # Compute maximum allowable step lengths
            fx = bound(x, dx)
            fs = bound(s, ds)
            fw = bound(w, dw)
            fz = bound(z, dz)
            fp = np.min([fx, fs])
            fd = np.min([fw, fz])
            fp = np.min([np.min(beta * fp), 1.0])
            fd = np.min([np.min(beta * fd), 1.0])
            
        # Take the step
        x = x + fp * dx
        s = s + fp * ds
        y = y + fd * dy
        w = w + fd * dw
        z = z + fd * dz
        gap = c.dot( x) - y.dot( b) + w.dot(u)
        if (display):
            print('{0} - {1}'.format(it, gap))
    return y
