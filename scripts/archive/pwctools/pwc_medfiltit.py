"""
Ported by Massimo Vassalli [http://mv.nanoscopy.eu massimo.vassalli@gmail.com]
"""
import numpy as np
from scipy.signal import medfilt

def pwc_medfiltit(y, W,display=False):
    # Performs running median filtering, iterated until convergence, using
    # window of size W. Requires MEDFILT1 from the Matlab signal processing
    # toolbox.
    #
    # Usage:
    # x = pwc_medfiltit(y, W)
    #
    # Input arguments:
    # - y          Original signal to denoise.
    # - W          Median filter window size, should be positive and odd.
    #
    # Output arguments:
    # - x          Denoised output signal.
    #
    # (c) Max Little, 2011. If you use this code for your research, please cite:
    # M.A. Little, Nick S. Jones (2011)
    # "Generalized Methods and Solvers for Noise Removal from Piecewise
    # Constant Signals: Part I - Background Theory"
    # Proceedings of the Royal Society A (in press)

    xold = y # Initialize iterated running medians
    
    # Iterate
    stop = False
    while (not stop):
        xnew = medfilt(xold,W);
        stop = np.all(xnew == xold);
        xold = xnew
        
    return xold
    
if __name__ == "__main__":
    y = [1 ,1.1, 0.9, 1.1, 0.95, 2.1, 1.95, 2.0, 2.05, 3.11, 2.99, 3.05, 3.0]
    print('Perform test')
    x = pwc_medfiltit(y,3)
    print(x)
