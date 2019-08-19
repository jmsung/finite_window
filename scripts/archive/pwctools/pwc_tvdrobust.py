"""
Ported by Massimo Vassalli [http://mv.nanoscopy.eu massimo.vassalli@gmail.com]
"""
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splin
import numpy.linalg as linalg
from lp_fnm import lp_fnm 

def pwc_tvdrobust(y, lamb=10.0, display=True, stoptol=1e-5, maxiter=60, full=False):
    # Performs robust discrete total variation denoising (TVD) using interior-
    # point linear programming. It minimizes the following discrete functional:
    #
    #  E=||y-x||_1+lambda*||Dx||_1,
    #
    # over the variable x, given the input signal y, according to the value
    # of the regularization parameter lambda >= 0. D is the first
    # difference matrix.
    #
    # Usage:
    # [x, E] = pwc_tvdrobust(y, lambda, display, stoptol, maxiter)
    #
    # Input arguments:
    # - y          Original signal to denoise.
    # - lambda     The non-negative regularization parameter.
    # - display    (Optional) Set to 0 to turn off progress display, 1 to turn
    #              on. If not specifed, defaults to progress display on.
    # - stoptol    (Optional) Precision as determined by duality gap tolerance,
    #              if not specified, defaults to 1e-5.
    # - maxiter    (Optional) Maximum interior-point iterations, if not
    #              specified defaults to 60.
    #
    # Output arguments:
    # - x          Denoised output signal.
    # - E          Objective functional at minimum.
    #
    # (c) Max Little, 2011. If you use this code for your research, please cite:
    # M.A. Little, Nick S. Jones (2011)
    # "Generalized Methods and Solvers for Noise Removal from Piecewise
    # Constant Signals: Part II - New Methods"
    # Proceedings of the Royal Society A (in press)
    
    y = np.array(y)
    N = len(y)
    
    # Construct sparse operator matrices
    D = sparse.lil_matrix((N,N))
    for i in range(N):
        D[i,i]=1.0
        if i<N-1:
            D[i,i+1]=-1.0
    D[-1,-1]=0
    
        
    if (display):
        print('Solving for lambda={0}\nIter# Gap'.format(lamb))
    
    # Set up robust TVD as L1-regression problem
    A = sparse.vstack((sparse.eye(N),-lamb*D),format='lil')
    b = sparse.vstack( ( y.reshape(N,1), sparse.lil_matrix((N,1))),format='lil' )
    
    # Minimize robust TV functional using interior-point linear programming.
    M = A.shape[0]
    u = np.ones(M)
    a = 0.5*u
    # Solves a linear program by the interior-point method:
    #x = -lp_fnm(A', -b', A'*a, u, a, stoptol, maxiter, display)';
    x = -lp_fnm(A.transpose(), -b.transpose(), A.transpose()*a, u, a)    
    if full:
        E = sum(abs(y-x)) + lamb*sum(abs(np.diff(data)))
        return x,E
    return x

if __name__ == "__main__":
    y = [1 ,1.1, 0.9, 1.1, 0.95, 2.1, 1.95, 2.0, 2.05, 3.11, 2.99, 3.05, 3.0]
    print('Perform test')
    x = pwc_tvdrobust(y)
    print(x)
