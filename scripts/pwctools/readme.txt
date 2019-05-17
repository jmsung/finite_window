This ZIP file contains software for Matlab and Python for performing noise removal
from 1D piecewise constant signals, as described and used in [1]. It includes
several example implementations of the methods described in [1],
including total variation denoising and robust total variation denoising by
interior-point algorithms, clustering using adaptive step size Euler integrators,
bilateral filtering, and jump penalization using greedy stepwise knot placement.
Ported to Python by Massimo Vassalli [http://mv.nanoscopy.eu massimo.vassalli@gmail.com].
Python requirements: PYTHON3, NUMPY, SCIPY (>=0.17.0 for tvdrobust to work)
No specific Python3 code was included, so that it should in principle run
with python2 by simply changing all print statements (not tested yet).

If you use this code for your research, please cite [1] below.

References:

[1] M.A. Little, N.S. Jones (2011)
Generalized Methods and Solvers for Noise Removal from Piecewise Constant Signals:
Parts I and II
Proceedings of the Royal Society A (in press)

In Matlab, type 'help (function)' for instructions for (function) below. ZIP file contents:

demo.m/demo.py
 - This shows an example application of DNA copy-number analysis with ten different
   algorithms using the functions described below. Run this first for hints about how
   to get the most from each algorithm.

pwc_tvdip.m/pwc_tvdip.py
 - Total variation denoising (TVD) using interior-point optimization.

tvdiplmax.m
 - Returns the largest useful value of the regularization parameter for the
   TVD function above.

pwc_tvdrobust.m/pwc_tvdrobust.py
 - Robust TVD using interior-point linear programming.

pwc_medfiltit.m/pwc_medfilt.py
 - Iterated median filtering. Requires the Matlab signal processing toolbox.

pwc_cluster.m/pwc_cluster.py
 - Clustering algorithms: K-means, mean-shift, likelihood mean shift, soft and
   biased versions of each. The solver is an adaptive step-size Euler integrator.

pwc_jumppenalty.m/pwc_jumppenalty.py
 - Jump penalization and robust jump penalization using greedy stepwise knot
   placement.

pwc_bilateral.m/pwc_bilateral.py
 - Bilateral and soft bilateral filter, solved using adaptive step-size Euler
   integration.
