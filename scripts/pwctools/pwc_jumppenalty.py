"""
Ported by Massimo Vassalli [http://mv.nanoscopy.eu massimo.vassalli@gmail.com]
"""
import numpy as np

def calcSolution(y,r1,square):
    N = len(y)
    xtest = np.zeros(N)
    if (square):
        xtest[0:r1[0]] = np.mean(y[0:r1[0]])
    else:
        xtest[0:r1[0]] = np.median(y[0:r1[0]])
    
    #for j in range(1,iiter+1):
    for j in range(1,len(r1)):
        if (square):
            xtest[r1[j-1]:r1[j]] = np.mean(y[r1[j-1]:r1[j]])
        else:
            xtest[r1[j-1]:r1[j]] = np. median(y[r1[j-1]:r1[j]])
    
    if (square):
        xtest[r1[-1]:] = np.mean(y[r1[-1]:])
    else:
        xtest[r1[-1]:] = np.median(y[r1[-1]:])
        
    return xtest


def pwc_jumppenalty(y, square=True, gamma=1.0, display=True, maxiter=50,full=False):
    # Performs PWC denoising of the input signal using least-squares or
    # least-absolute (robust) forward jump penalization. It attempts to
    # minimize the discrete functional:
    #
    #  E=||y-x||_p+gamma*||Dx||_0,
    # 
    # where ||.||_0 is the L0 norm, here used to count the number of jumps in x.
    # The stopping criteria is at the first minima of H encountered.
    #
    # Usage:
    # x = pwc_jumppenalty(y, square, gamma, maxiter)
    #
    # Input arguments:
    # - y          Original signal to denoise of length N.
    # - square     Set to 1 to perform least-squares fitting (p=2 in the global
    #              functional), or 0 to perform least-absolute (robust) fitting
    #              (p=1).
    # - gamma      Positive regularization parameter.
    # - display    (Optional) Set to 0 to turn off progress display, 1 to turn
    #              on. If not specifed, defaults to progress display on.
    # - maxiter    (Optional) Maximum number of iterations. If not specified,
    #              defaults to 50.
    #
    # Output arguments:
    # - x          Denoised output signal.
    # - H          Global penalized functional value given the output signal.
    #
    # (c) Max Little, 2011. If you use this code for your research, please cite:
    # M.A. Little, Nick S. Jones (2011)
    # "Generalized Methods and Solvers for Noise Removal from Piecewise
    # Constant Signals: Part II - New Methods"
    # Proceedings of the Royal Society A (in press)

    y = np.array(y)
    N = len(y)
    
    r = []
    Eold = np.Inf

    if (display):
        print('Iter# Global functional')

    # Iterate
    iiter = 0;
    while (iiter < maxiter):
    
        if (display):
            print('{0} {1}'.format(iiter,Eold));
    
        # Greedy scan for new knot location
        NLL = np.zeros(N);
        for i in range(N):
            
            # Append new location
            r1 = r.copy()
            r1.append(i)
            r1.sort()
            
            xtest = calcSolution(y,r1,square)
            # Compute likelihood
            if (square):
                NLL[i] = 0.5*sum((xtest-y)**2)
            else:
                NLL[i] = sum(abs(xtest-y))
                
        # Choose new location that minimizes the likelihood term
        i =np.argmin(NLL);
        # Add to knot locations
        r.append(i)
        r.sort()
    
        # Re-compute solution at this iteration
        xnew = calcSolution(y,r,square)        
        
        # Compute global functional value at current solution
        if (square):
            Enew = 0.5*sum((xnew-y)**2) + gamma*iiter;
        else:
            Enew = sum(abs(xnew-y)) + gamma*iiter;

        # Detect local minima in global functional
        if (Eold < Enew):
            if (display):
                print('Converged in {0} iterations'.format(iiter));
            break;
        
        Eold = Enew;
        iiter += 1

    if (display):
        if (iiter == maxiter):
            print('Maximum iterations exceeded')
    if full:
        return xnew,Enew
    return xnew
    
if __name__ == "__main__":
    y = [1 ,1.1, 0.9, 1.1, 0.95, 2.1, 1.95, 2.0, 2.05, 3.11, 2.99, 3.05, 3.0]
    print('Perform test')
    x = pwc_jumppenalty(y)
    print(x)
