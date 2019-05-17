"""
Ported by Massimo Vassalli [http://mv.nanoscopy.eu massimo.vassalli@gmail.com]

This file is a helper for the use of the included functions, ported from 
the original example set. Example DNA data was originally distributed with
the Matlab implementation
"""

# Load DNA copy-number data
import numpy as np
import matplotlib.pylab as plt

from pwc_medfiltit import pwc_medfiltit
from pwc_cluster import pwc_cluster
from pwc_jumppenalty import pwc_jumppenalty
from pwc_bilateral import pwc_bilateral
from pwc_tvdip import pwc_tvdip
from pwc_tvdrobust import pwc_tvdrobust

y = np.loadtxt('dnagwas.txt');
N = len(y);
L = 10; #number of (implemented) plots
x = np.zeros((N,L))
name = []

# Iterated median filter
name.append('Iterated medians')
x[:,0] = pwc_medfiltit(y,15)

# Classical K-means with K=5
name.append('Classical K-means')
x[:,1] = pwc_cluster(y,5);

# Soft K-means with Gaussian kernel and K=3
name.append('Soft K-means')
x[:,2] = pwc_cluster(y,3,soft=True,beta=80.0,biased=True);

# Likelihood mean-shift with hard kernel
name.append('Likelihood mean-shift')
x[:,3] = pwc_cluster(y,beta=0.1,biased=True);

# Soft mean-shift with Gaussian kernel
name.append('Soft mean-shift')
x[:,4] = pwc_cluster(y,soft=True,beta=800.0,biased=False);

# Total variation denoising
name.append('Total variation denoising')
x[:,5] = pwc_tvdip(y,[1.0])[:,0]

# Robust total variation denoising
name.append('Robust TVD')
x[:,6] = pwc_tvdrobust(y);

# Jump penalization
name.append('Jump penalty')
x[:,7] = pwc_jumppenalty(y);

# Robust jump penalization
name.append('Robust jump penalty')
x[:,8] = pwc_jumppenalty(y,square=False);

# Bilateral filter with Gaussian kernel
name.append('Bilateral filter')
x[:,9] = pwc_bilateral(y);

# Plots
plt.figure()
c = [0.6, 0.6, 0.6]
R = np.ceil(np.sqrt(L*1.0)-2);
C = np.ceil(1.0*L/R);


for i in range(L):
    plt.subplot(R,C,i+1)
    plt.plot(y,color=c)
    plt.plot(x[:,i],'b-')
    plt.title(name[i])
    
plt.show()
