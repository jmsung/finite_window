"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Created by Jongmin Sung (jongmin.sung@gmail.com)

Simulation to test the finite_window analysis 

class Data() 
- path, name, load(), list[], n_movie, movies = [Movie()], plot(), analysis(), spot_size, frame_rate, 
- path, name, n_row, n_col, n_frame, pixel_size, 
- background, spots = [], molecules = [Molecule()]

class Molecule()
- position (row, col), intensity 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from __future__ import division, print_function, absolute_import
import numpy as np
from numpy.random import rand, randn
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path  
import os
import shutil
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.signal import argrelextrema
from scipy.stats import sem
from tifffile import TiffFile
from imreg_dft.imreg import translation
from skimage.feature import peak_local_max
from skimage.filters import threshold_local
from hmmlearn import hmm
from inspect import currentframe, getframeinfo
fname = getframeinfo(currentframe()).filename # current file name
current_dir = Path(fname).resolve().parent
data_dir = current_dir.parent/'data' 

# User input ----------------------------------------------------------------
directory = data_dir/'simulation'

# Parameters for simulation
n_data = 1
n_peak = 10000
n_frames = range(100, 101, 50) 
frame_b = 5
frame_u = 50
duty_ratio = frame_b/(frame_b+frame_u)
mean_b = 1000
mean_u = 100
width_b = 1
width_u = 1
noise_b = 10
noise_u = 10

# ---------------------------------------------------------------------------

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "y", "1")

def is_inlier(I, m=4):
    dev = np.abs(I - np.median(I))
    mdev = np.median(dev)
    s = dev/mdev if mdev else 0.
    return s < m

def running_avg(x, n):
    m = int((n-1)/2)
    y =  np.convolve(x, np.ones((n,))/n, mode='valid') 
    z = [np.round(i) for i in y]
    k = np.asarray(z[:1]*m + z + z[-1:]*m, dtype=int)
    return k

def sum_two_gaussian(x, m1, s1, f1, m2, s2, f2):
#    x = running_avg(x,3)
    return abs(f1)*np.exp(-(x-m1)**2/(2*s1**2)) + abs(f2)*np.exp(-(x-m2)**2/(2*s2**2))

def sum_two_lorentzian(x, m1, s1, f1, m2, s2, f2):
    return abs(f1)/(1+(x-m1)**2/s1**2) + abs(f2)/(1+(x-m2)**2/s2**2)

def sum_gaussian_lorentzian(x, m1, s1, f1, m2, s2, f2):
    return abs(f1)*np.exp(-(x-m1)**2/(2*s1**2)) + abs(f2)/(1+(x-m2)**2/s2**2)

def get_mean(T, t, cl):
    tau = np.mean(t)
    return tau
#    if cl == 2:
#        return tau
#    else:
#        for i in range(20):
#            tau = np.mean(t) + T/(np.exp(T/tau)-1)
#        return tau

def exp_icdf(k, T, t, cl):
    return np.exp(-k*t)

def exp_pdf(k, T, t, cl):
    return k*np.exp(-k*t)

def get_icdf(t):
    icdf = []
    x = range(int(max(t)+1))
    for i in x:
        icdf.append(sum(i<=t)/len(t)) 
    return x, icdf

def icdf(k, T, t, cl):
    if cl == 2:
        A = k*T-1
        return 1 - (np.exp(-k*t)*(k*t-A)+A)/(np.exp(-k*T)+A)        
    else:
        return (np.exp(-k*t)-np.exp(-k*T))/(1-np.exp(-k*T))

def pdf(k, T, t, cl):
    if cl == 2:
        return (k*T-k*t)/(k*T-1+np.exp(-k*T))*k*np.exp(-k*t)  
    else:        
        return k*np.exp(-k*t)/(1-np.exp(-k*T))

def LL(k, T, t, cl):     
    k = abs(k)
    if cl == 2:
        return np.sum(np.log(k*T-k*t)-np.log(k*T-1+np.exp(-k*T))+np.log(k)-k*t) 
    else:        
        return np.sum(np.log(k)-k*t-np.log(1-np.exp(-k*T)))


def MLE(T, t, cl):
    fun = lambda *args: -LL(*args)
    p0 = [1/np.mean(t)]
    result = minimize(fun, p0, method='SLSQP', args=(T, t, cl)) 
    return abs(1/result["x"][0])

def Info(k, T, t, cl):
    dk = k/100
    return abs(LL(k+dk,T,t,cl)+LL(k-dk,T,t,cl)-2*LL(k,T,t,cl))/dk**2

def get_weighted_mean(m1, s1, m2, s2, m3, s3):
    w1 = 1/s1**2
    w2 = 1/s2**2
    w3 = 1/s3**2
    weighted_mean = (w1*m1+w2*m2+w3*m3)/(w1+w2+w3) 
    weighted_error = (w1+w2+w3)**-0.5
    return weighted_mean, weighted_error

class Movie:
    def __init__(self, path):
        self.path = path
        self.dir = path.parent
        self.name = path.name

    def read_info(self):
        # Read info.txt
        self.info = {}
        with open(Path(self.dir/'info.txt')) as f:
            for line in f:
                line = line.replace(" ", "") # remove white space
                if line == '\n': # skip empty line
                    continue
                (key, value) = line.rstrip().split("=")
                self.info[key] = value

        # Parameters for analysis 
        self.dt = float(self.info['time_interval'])
        self.save_trace = int(self.info['save_trace'])
        self.intensity_min_cutoff = float(self.info['intensity_min_cutoff']) 
        self.intensity_max_cutoff = float(self.info['intensity_max_cutoff']) 
        self.HMM_RMSD_cutoff = float(self.info['HMM_RMSD_cutoff']) 
        self.HMM_unbound_cutoff = float(self.info['HMM_unbound_cutoff']) 
        self.HMM_bound_cutoff = float(self.info['HMM_bound_cutoff']) 


    def simulate_peak(self, n_frame, n_peak):
        self.n_frame = n_frame
        self.window = self.n_frame*self.dt
        self.n_peak = n_peak
        self.time_b = frame_b*self.dt
        self.time_u = frame_u*self.dt

        # Simulate peak trace based on the parameters`
        Z = np.zeros((self.n_peak, self.n_frame), dtype='int')
        I = np.zeros((self.n_peak, self.n_frame), dtype='float')

        for i in range(self.n_peak):
            # Initial condition 
            if rand() > duty_ratio: 
                Z[i][0] = 0
                I_u = mean_u + width_u*randn() 
                I[i][0] = I_u + noise_u*randn() 
            else:
                Z[i][0] = 1
                I_b = mean_b + width_b*randn()
                I[i][0] = I_b + noise_b*randn()  

            # Transition 
            for j in range(self.n_frame-1):
                if Z[i][j] == 0:
                    if rand() > 1 - np.exp(-1/frame_u):
                        Z[i][j+1] = 0
                        I[i][j+1] = I_u + noise_u*randn()
                    else:
                        Z[i][j+1] = 1
                        I_b = mean_b + width_b*randn()
                        I[i][j+1] = I_b + noise_b*randn()
                else:
                    if rand() > 1 - np.exp(-1/frame_b):
                        Z[i][j+1] = 1        
                        I[i][j+1] = I_b + noise_b*randn()
                    else:
                        Z[i][j+1] = 0
                        I_u = mean_u + width_u*randn()
                        I[i][j+1] = I_u + noise_u*randn()
                                                       
        self.peak_trace = I
        self.state = Z


    # Evaluate quality of signal
    def find_spot(self):
        # Find inliers with I_max and I_min
        self.peak_min = np.min(self.peak_trace, axis=1)
        self.peak_max = np.max(self.peak_trace, axis=1)
        self.is_peak_min_inlier = is_inlier(self.peak_min, self.intensity_min_cutoff)
        self.is_peak_max_inlier = is_inlier(self.peak_max, self.intensity_max_cutoff)
        self.is_peak_inlier = self.is_peak_min_inlier & self.is_peak_max_inlier

        self.n_spot = sum(self.is_peak_inlier)
        self.trace = self.peak_trace[self.is_peak_inlier]

        # Histogram to find intensity distributions for unbound and bound states
        n, bins = np.histogram(self.trace.flatten(), bins=100, density=False)
        x = (bins[1:]+bins[:-1])/2
        m1 = 0.8*x[0] + 0.2*x[-1]
        s1 = 0.1*(x[-1] - x[0]) 
        n1 = max(n)
        m2 = 0.5*x[0] + 0.5*x[-1]
        s2 = 0.2*(x[-1] - x[0])
        n2 = n1/2
        self.I_param, cov = curve_fit(sum_two_gaussian, x, n, p0=[m1, s1, n1, m2, s2, n2])
        self.I_hist = n
        self.I_x = x


    # Fit traces
    def fit_spot(self):
        self.trace_fit = np.zeros((self.n_spot, self.n_frame))
        self.state = np.zeros((self.n_spot, self.n_frame))        
        self.rmsd = np.zeros(self.n_spot)
        self.I_u = np.zeros(self.n_spot)
        self.I_b = np.zeros(self.n_spot)

        # Fit the time trace using HMM    
        for i, trace in enumerate(self.trace):
            X = trace.reshape(len(trace), 1) 
          
            # Set a new model for traidning
            param=set(X.ravel())
            remodel = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)        
        
            # Set initial parameters for training
            remodel.startprob_ = np.array([self.I_param[2]/(self.I_param[2]+self.I_param[5]), 
                                           self.I_param[5]/(self.I_param[2]+self.I_param[5])])
            remodel.transmat_ = np.array([[0.99, 0.01], 
                                          [0.10, 0.90]])
            remodel.means_ = np.array([self.I_param[0], self.I_param[3]])  
            remodel.covars_ = np.array([[[self.I_param[1]]],
                                        [[self.I_param[4]]]])
           
            # Estimate model parameters (training)
            remodel.fit(X)

            # Find most likely state sequence corresponding to X
            Z = remodel.predict(X)

            # Reorder state number such that X[Z=0] < X[Z=1] 
            if remodel.means_[0] > remodel.means_[1]:
                Z = 1 - Z
                remodel.means_ = remodel.means_[::-1]

            # Intensity trace fit     
            self.state[i] = np.array(Z) 
            self.trace_fit[i] = (1-Z)*remodel.means_[0] + Z*remodel.means_[1]     
            self.rmsd[i] = (np.mean((self.trace_fit[i] - trace)**2))**0.5           

            # Mean intensity of the two states
            self.I_u[i] = remodel.means_[0]
            self.I_b[i] = remodel.means_[1]

        # Find inliners and exclude outliers
        self.is_rmsd_inlier = self.rmsd < np.median(self.rmsd)*self.HMM_RMSD_cutoff
        self.is_I_u_inlier = is_inlier(self.I_u, self.HMM_unbound_cutoff)
        self.is_I_b_inlier = is_inlier(self.I_b, self.HMM_bound_cutoff)
        self.is_trace_inlier = self.is_rmsd_inlier & self.is_I_u_inlier & self.is_I_b_inlier

        # Save inlier traces
        self.state_inlier = self.state[self.is_trace_inlier]
        self.trace_inlier = self.trace[self.is_trace_inlier]
        self.I_u_inlier = self.I_u[self.is_trace_inlier]        
        self.I_b_inlier = self.I_b[self.is_trace_inlier]
        self.rmsd_inlier = self.rmsd[self.is_trace_inlier]

#        print('Found', self.n_peak, 'peaks. ')     
#        print('Rejected', self.n_peak - len(self.rmsd_inlier), 'outliers.')   



    def find_event(self):
        self.dwell_1 = [] # Bound, class 1 (pre-existing)
        self.dwell_2 = [] # Bound, class 2 (complete)
        self.dwell_3 = [] # Bound, class 3 (incomplete)
        self.wait_1 = [] # Unbound, class 1 (pre-existing)
        self.wait_2 = [] # Unbound, class 2 (complete)
        self.wait_3 = [] # Unbound, class 3 (incomplete)

        for _, state in enumerate(self.state_inlier):
            tb = [] # Frame at binding
            tu = [] # Frame at unbinding
    
            # Find binding and unbinding moment
            for i in range(self.n_frame-1):
                if state[i] == 0 and state[i+1] == 1: # binding
                    tb.append(i) 
                elif state[i] == 1 and state[i+1] == 0: # unbinding
                    tu.append(i) 
                else:
                    pass

            # Cases 
            if len(tb) + len(tu) == 0: # n_event = 0
                continue
            elif len(tb) + len(tu) == 1: # n_event = 1
                if len(tb) == 1: # One binding event
                    self.wait_1.append(tb[0]+1)
                    self.dwell_3.append(self.n_frame-tb[-1]-1)
                else: # One unbinding event 
                    self.dwell_1.append(tu[0]+1)
                    self.wait_3.append(self.n_frame-tu[-1]-1)
            else: # n_event > 1 
                # First event is w1 or d1
                if state[0] == 0: # Unbound state at the beginning
                    self.wait_1.append(tb[0]+1)
                else: # Bound state at the beginning
                    self.dwell_1.append(tu[0]+1)

                # Last event is w3 or d3
                if state[-1] == 0: # Unbound state at the end
                    self.wait_3.append(self.n_frame-tu[-1]-1)
                else: # Bound state at the end
                    self.dwell_3.append(self.n_frame-tb[-1]-1)

                # All the rests are w2 or d2
                t = tb + tu # Concatenate and sort in order 
                t.sort()
                dt = [t[i+1]-t[i] for i in range(len(t)-1)]
                dt_odd = dt[0::2]
                dt_even = dt[1::2]

                if state[0] == 0: # Odd events are d2, event events are w2
                    self.dwell_2.extend(dt_odd)
                    self.wait_2.extend(dt_even)
                else: # Odd events are w2, event events are d2
                    self.wait_2.extend(dt_odd)
                    self.dwell_2.extend(dt_even)                


    def  exclude_short(self):

        # Offset 1.5 frame
        self.offset = 1.5

        self.dwell_1 = np.array(self.dwell_1)-self.offset
        self.dwell_2 = np.array(self.dwell_2)-self.offset
        self.dwell_3 = np.array(self.dwell_3)-self.offset

        self.wait_1 = np.array(self.wait_1)-self.offset
        self.wait_2 = np.array(self.wait_2)-self.offset
        self.wait_3 = np.array(self.wait_3)-self.offset

        # Exclude short frames and convert unit in sec 
        self.dwell_1 = self.dwell_1[self.dwell_1>0]*self.dt
        self.dwell_2 = self.dwell_2[self.dwell_2>0]*self.dt
        self.dwell_3 = self.dwell_3[self.dwell_3>0]*self.dt

        self.wait_1 = self.wait_1[self.wait_1>0]*self.dt
        self.wait_2 = self.wait_2[self.wait_2>0]*self.dt
        self.wait_3 = self.wait_3[self.wait_3>0]*self.dt

   
    def estimate_time(self):
        # Mean estimation 
        self.mean_dwell_1 = get_mean(self.window, self.dwell_1, 1)
        self.mean_dwell_2 = get_mean(self.window, self.dwell_2, 2)
        self.mean_dwell_3 = get_mean(self.window, self.dwell_3, 3)

        self.mean_wait_1 = get_mean(self.window, self.wait_1, 1)
        self.mean_wait_2 = get_mean(self.window, self.wait_2, 2)
        self.mean_wait_3 = get_mean(self.window, self.wait_3, 3)

        # MLE estimation 
        self.MLE_dwell_1 = MLE(self.window, self.dwell_1, 1) 
        self.MLE_dwell_2 = MLE(self.window, self.dwell_2, 2) 
        self.MLE_dwell_3 = MLE(self.window, self.dwell_3, 3) 

        self.MLE_wait_1 = MLE(self.window, self.wait_1, 1) 
        self.MLE_wait_2 = MLE(self.window, self.wait_2, 2) 
        self.MLE_wait_3 = MLE(self.window, self.wait_3, 3) 
                    
        # MLE error from Fisher Information
        self.Error_dwell_1 = Info(self.MLE_dwell_1, self.window, self.dwell_1, 1)**-0.5 
        self.Error_dwell_2 = Info(self.MLE_dwell_2, self.window, self.dwell_2, 2)**-0.5         
        self.Error_dwell_3 = Info(self.MLE_dwell_3, self.window, self.dwell_3, 3)**-0.5 

        self.Error_wait_1 = Info(self.MLE_wait_1, self.window, self.wait_1, 1)**-0.5 
        self.Error_wait_2 = Info(self.MLE_wait_2, self.window, self.wait_2, 2)**-0.5 
        self.Error_wait_3 = Info(self.MLE_wait_3, self.window, self.wait_3, 3)**-0.5 

        # Weighted MLE
        self.MLE_dwell, self.Error_dwell = get_weighted_mean(self.MLE_dwell_1, self.Error_dwell_1, 
                                                             self.MLE_dwell_2, self.Error_dwell_2, 
                                                             self.MLE_dwell_3, self.Error_dwell_3)

        self.MLE_wait, self.Error_wait = get_weighted_mean(self.MLE_wait_1, self.Error_wait_1, 
                                                            self.MLE_wait_2, self.Error_wait_2, 
                                                            self.MLE_wait_3, self.Error_wait_3)

    def save_result(self):
        # Write the result to an output file
        with open(Path(self.dir/'result.txt'), "w") as f:
            f.write('directory = %s \n' %(self.dir))
            f.write('name = %s \n' %(self.name))
            f.write('time interval = %.2f [s] \n' %(self.dt))
            f.write('number of frame = %d \n' %(self.n_frame))
            f.write('number of spots = %d \n' %(len(self.rmsd_inlier)))
            f.write('dwell time (class 1) = %.2f +/- %.2f [s] (N = %d) \n' %(self.MLE_dwell_1, self.Error_dwell_1, len(self.dwell_1)))
            f.write('dwell time (class 2) = %.2f +/- %.2f [s] (N = %d) \n' %(self.MLE_dwell_2, self.Error_dwell_2, len(self.dwell_2)))
            f.write('dwell time (class 3) = %.2f +/- %.2f [s] (N = %d) \n' %(self.MLE_dwell_3, self.Error_dwell_3, len(self.dwell_3)))
            f.write('dwell time (combined) = %.2f +/- %.2f [s] (N = %d) \n' %(self.MLE_dwell, self.Error_dwell, len(self.dwell_1)+len(self.dwell_2)+len(self.dwell_3)))
            f.write('wait time (class 1) = %.2f +/- %.2f [s] (N = %d) \n' %(self.MLE_wait_1, self.Error_wait_1, len(self.wait_1)))
            f.write('wait time (class 2) = %.2f +/- %.2f [s] (N = %d) \n' %(self.MLE_wait_2, self.Error_wait_2, len(self.wait_2)))
            f.write('wait time (class 3) = %.2f +/- %.2f [s] (N = %d) \n' %(self.MLE_wait_3, self.Error_wait_3, len(self.wait_3)))
            f.write('wait time (combined) = %.2f +/- %.2f [s] (N = %d) \n' %(self.MLE_wait, self.Error_wait, len(self.wait_1)+len(self.wait_2)+len(self.wait_3)))

    def plot0_clean(self):
        # clean all existing png files in the folder
        files = os.listdir(self.dir)    
        for file in files:
            if file.endswith('png'):
                os.remove(self.dir/file)    


    def plot5_spot_fit(self):
        spot_trace = self.trace.flatten()
        x_fit = np.linspace(min(self.I_x), max(self.I_x), 1000)
        y_fit = sum_two_gaussian(x_fit, *self.I_param)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(20, 10), ncols=2, nrows=2, dpi=300)     
  
#        is_max = argrelextrema(running_avg(np.log(self.I_hist), 5), np.greater)
#        print(is_max)

        ax1.step(self.I_x, self.I_hist, where='mid', c='k', lw=2)
#        ax1.scatter(self.I_x[is_max], self.I_hist[is_max], lw=2, s=100, facecolors='none', edgecolors='r')
#        ax4.plot(x_fit, y_fit, 'r')    
        ax1.set_yscale('log')      
        ax1.set_xlabel('Intensity')
        ax1.set_ylabel('Counts')
        ax1.set_title('Intensity of entire traces')   

        bins = np.linspace(min(self.rmsd), max(self.rmsd), 50)          
        ax2.hist(self.rmsd, bins = bins, histtype='step', lw=2, color='b')   
        ax2.hist(self.rmsd[self.is_rmsd_inlier], bins = bins, histtype='step', lw=2, color='r')   
        ax2.set_title('RMSD of fitting (HMM)')
        ax2.set_xlabel('RMSD')
        ax2.set_ylabel('Counts')

        bins = np.linspace(min(self.I_u), max(self.I_u), 50)  
        ax3.hist(self.I_u, bins = bins, histtype='step', lw=2, color='b')      
        ax3.hist(self.I_u[self.is_I_u_inlier], bins = bins, histtype='step', lw=2, color='r')    
        ax3.set_title('Intensity unbound (HMM)')
        ax3.set_xlabel('Intensity')
        ax3.set_ylabel('Counts')
 
        bins = np.linspace(min(self.I_b), max(self.I_b), 50)  
        ax4.hist(self.I_b, bins = bins, histtype='step', lw=2, color='b')      
        ax4.hist(self.I_b[self.is_I_b_inlier], bins = bins, histtype='step', lw=2, color='r')      
        ax4.set_title('Intensity bound (HMM)')
        ax4.set_xlabel('Intensity')
        ax4.set_ylabel('Counts')

        fig.savefig(self.dir/'plot5_spot_fit.png')   
        plt.close(fig)


    def plot6_dwell_pdf(self):

        t1 = self.dwell_1
        t2 = self.dwell_2
        t3 = self.dwell_3
        t_max = max(t1.tolist()+t2.tolist()+t3.tolist())

        n_bin = 20

        if t_max > n_bin*self.dt: 
            interval = np.ceil(t_max/self.dt/n_bin)*self.dt
            bins = np.arange(0, interval*(n_bin+1), interval)
        else:
            interval = self.dt
            bins = np.arange(0, t_max+interval, interval) 

        x_fit = np.linspace(0, t_max+interval/2, 100)

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(figsize=(20, 10), ncols=3, nrows=2, dpi=300)
  
        ax1.hist(t1, bins=bins, histtype='step', lw=1, color='k', density=True)
        ax1.plot(x_fit, pdf(1/self.time_b, self.window, x_fit, 1), 'k', lw=1)    
        ax1.plot(x_fit, exp_pdf(1/self.mean_dwell_1, self.window, x_fit, 1), 'b', lw=1) 
        ax1.plot(x_fit, pdf(1/self.MLE_dwell_1, self.window, x_fit, 1), 'r', lw=1)  
        ax1.set_ylabel('Probability density')
        ax1.set_title('Class 1 (N = %d), True = %.2f [s]' %(len(t1), self.time_b))

        ax2.hist(t2, bins=bins, histtype='step', lw=1, color='k', density=True)
        ax2.plot(x_fit, pdf(1/self.time_b, self.window, x_fit, 2), 'k', lw=1)    
        ax2.plot(x_fit, exp_pdf(1/self.mean_dwell_2, self.window, x_fit, 2), 'b', lw=1) 
        ax2.plot(x_fit, pdf(1/self.MLE_dwell_2, self.window, x_fit, 2), 'r', lw=1)                   
        ax2.set_ylabel('Probability density')
        ax2.set_title('Class 2 (N = %d), Combined Exp_Finite = %.2f +/- %.2f [s]' %(len(t2), self.MLE_dwell, self.Error_dwell))
  
        ax3.hist(t3, bins=bins, histtype='step', lw=1, color='k', density=True)
        ax3.plot(x_fit, pdf(1/self.time_b, self.window, x_fit, 3), 'k', lw=1)    
        ax3.plot(x_fit, exp_pdf(1/self.mean_dwell_3, self.window, x_fit, 3), 'b', lw=1) 
        ax3.plot(x_fit, pdf(1/self.MLE_dwell_3, self.window, x_fit, 3), 'r', lw=1)        
        ax3.set_ylabel('Probability density')
        ax3.set_title('Class 3 (N = %d), True (K), Exp (B), Exp_Finite (R)' %(len(t3)))
  
        ax4.hist(t1, bins=bins, histtype='step', lw=1, color='k', density=True)
        ax4.plot(x_fit, pdf(1/self.time_b, self.window, x_fit, 1), 'k', lw=1)    
        ax4.plot(x_fit, exp_pdf(1/self.mean_dwell_1, self.window, x_fit, 1), 'b', lw=1) 
        ax4.plot(x_fit, pdf(1/self.MLE_dwell_1, self.window, x_fit, 1), 'r', lw=1)      
        ax4.set_yscale('log')
        ax4.set_xlabel('Dwell Time [s]')
        ax4.set_ylabel('Probability density')
        ax4.set_title('Exp = %.2f [s], Exp_Finite = %.2f +/- %.2f [s]' %(self.mean_dwell_1, self.MLE_dwell_1, self.Error_dwell_1))        
  
        ax5.hist(t2, bins=bins, histtype='step', lw=1, color='k', density=True)
        ax5.plot(x_fit, pdf(1/self.time_b, self.window, x_fit, 2), 'k', lw=1)    
        ax5.plot(x_fit, exp_pdf(1/self.mean_dwell_2, self.window, x_fit, 2), 'b', lw=1) 
        ax5.plot(x_fit, pdf(1/self.MLE_dwell_2, self.window, x_fit, 2), 'r', lw=1)          
        ax5.set_yscale('log')
        ax5.set_xlabel('Dwell Time [s]')
        ax5.set_ylabel('Probability density')
        ax5.set_title('Exp = %.2f [s], Exp_Finite = %.2f +/- %.2f [s]' %(self.mean_dwell_2, self.MLE_dwell_2, self.Error_dwell_2))  
   
        ax6.hist(t3, bins=bins, histtype='step', lw=1, color='k', density=True)
        ax6.plot(x_fit, pdf(1/self.time_b, self.window, x_fit, 3), 'k', lw=1)    
        ax6.plot(x_fit, exp_pdf(1/self.mean_dwell_3, self.window, x_fit, 3), 'b', lw=1) 
        ax6.plot(x_fit, pdf(1/self.MLE_dwell_3, self.window, x_fit, 3), 'r', lw=1)         
        ax6.set_yscale('log')
        ax6.set_xlabel('Dwell Time [s]')
        ax6.set_ylabel('Probability density')
        ax6.set_title('Exp = %.2f [s], Exp_Finite = %.2f +/- %.2f [s]' %(self.mean_dwell_3, self.MLE_dwell_3, self.Error_dwell_3))  

        fig.savefig(self.dir/'plot6_dwell_pdf.png')   
        plt.close(fig)


    def plot7_wait_pdf(self):

        t1 = self.wait_1
        t2 = self.wait_2
        t3 = self.wait_3
        t_max = max(t1.tolist()+t2.tolist()+t3.tolist())

        n_bin = 20

        if t_max > n_bin*self.dt: 
            interval = np.ceil(t_max/self.dt/n_bin)*self.dt
            bins = np.arange(0, interval*(n_bin+1), interval)
        else:
            interval = self.dt
            bins = np.arange(0, t_max+interval, interval) 

        x_fit = np.linspace(0, t_max+interval/2, 100)

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(figsize=(20, 10), ncols=3, nrows=2, dpi=300)   
   
        ax1.hist(t1, bins=bins, histtype='step', lw=1, color='k', density=True)
        ax1.plot(x_fit, pdf(1/self.time_u, self.window, x_fit, 1), 'k', lw=1)    
        ax1.plot(x_fit, exp_pdf(1/self.mean_wait_1, self.window, x_fit, 1), 'b', lw=1) 
        ax1.plot(x_fit, pdf(1/self.MLE_wait_1, self.window, x_fit, 1), 'r', lw=1)  
        ax1.set_ylabel('Probability density')
        ax1.set_title('Class 1 (N = %d), True = %.2f [s]' %(len(t1), self.time_u))

        ax2.hist(t2, bins=bins, histtype='step', lw=1, color='k', density=True)
        ax2.plot(x_fit, pdf(1/self.time_u, self.window, x_fit, 2), 'k', lw=1)    
        ax2.plot(x_fit, exp_pdf(1/self.mean_wait_2, self.window, x_fit, 2), 'b', lw=1) 
        ax2.plot(x_fit, pdf(1/self.MLE_wait_2, self.window, x_fit, 2), 'r', lw=1)                   
        ax2.set_ylabel('Probability density')
        ax2.set_title('Class 2 (N = %d), Combined Exp_Finite = %.2f +/- %.2f [s]' %(len(t2), self.MLE_wait, self.Error_wait))
  
        ax3.hist(t3, bins=bins, histtype='step', lw=1, color='k', density=True)
        ax3.plot(x_fit, pdf(1/self.time_u, self.window, x_fit, 3), 'k', lw=1)    
        ax3.plot(x_fit, exp_pdf(1/self.mean_wait_3, self.window, x_fit, 3), 'b', lw=1) 
        ax3.plot(x_fit, pdf(1/self.MLE_wait_3, self.window, x_fit, 3), 'r', lw=1)        
        ax3.set_ylabel('Probability density')
        ax3.set_title('Class 3 (N = %d), True (K), Exp (B), Exp_Finite (R)' %(len(t3)))

        ax4.hist(t1, bins=bins, histtype='step', lw=1, color='k', density=True)
        ax4.plot(x_fit, pdf(1/self.time_u, self.window, x_fit, 1), 'k', lw=1)    
        ax4.plot(x_fit, exp_pdf(1/self.mean_wait_1, self.window, x_fit, 1), 'b', lw=1) 
        ax4.plot(x_fit, pdf(1/self.MLE_wait_1, self.window, x_fit, 1), 'r', lw=1)      
        ax4.set_yscale('log')
        ax4.set_xlabel('Wait Time [s]')
        ax4.set_ylabel('Probability density')
        ax4.set_title('Exp = %.2f [s], Exp_Finite = %.2f +/- %.2f [s]' %(self.mean_wait_1, self.MLE_wait_1, self.Error_wait_1))        
  
        ax5.hist(t2, bins=bins, histtype='step', lw=1, color='k', density=True)
        ax5.plot(x_fit, pdf(1/self.time_u, self.window, x_fit, 2), 'k', lw=1)    
        ax5.plot(x_fit, exp_pdf(1/self.mean_wait_2, self.window, x_fit, 2), 'b', lw=1) 
        ax5.plot(x_fit, pdf(1/self.MLE_wait_2, self.window, x_fit, 2), 'r', lw=1)          
        ax5.set_yscale('log')
        ax5.set_xlabel('Wait Time [s]')
        ax5.set_ylabel('Probability density')
        ax5.set_title('Exp = %.2f [s], Exp_Finite = %.2f +/- %.2f [s]' %(self.mean_wait_2, self.MLE_wait_2, self.Error_wait_2))  
   
        ax6.hist(t3, bins=bins, histtype='step', lw=1, color='k', density=True)
        ax6.plot(x_fit, pdf(1/self.time_u, self.window, x_fit, 3), 'k', lw=1)    
        ax6.plot(x_fit, exp_pdf(1/self.mean_wait_3, self.window, x_fit, 3), 'b', lw=1) 
        ax6.plot(x_fit, pdf(1/self.MLE_wait_3, self.window, x_fit, 3), 'r', lw=1)         
        ax6.set_yscale('log')
        ax6.set_xlabel('Wait Time [s]')
        ax6.set_ylabel('Probability density')
        ax6.set_title('Exp = %.2f [s], Exp_Finite = %.2f +/- %.2f [s]' %(self.mean_wait_3, self.MLE_wait_3, self.Error_wait_3))  
        fig.savefig(self.dir/'plot7_wait_pdf.png')   
        plt.close(fig)


    def plot8_dwell_icdf(self):

        t1 = self.dwell_1
        t2 = self.dwell_2
        t3 = self.dwell_3
        t_max = max(t1.tolist()+t2.tolist()+t3.tolist())

        bins = np.arange(0, t_max+1, 1)
        x_fit = np.linspace(0, t_max+1, 100)

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(figsize=(20, 10), ncols=3, nrows=2, dpi=300)

        x1, n1 = get_icdf(t1)
        ax1.step(x1, n1, where='mid', c='k', lw=1)
        ax1.plot(x_fit, icdf(1/self.time_b, self.window, x_fit, 1), 'k', lw=1)   
        ax1.plot(x_fit, exp_icdf(1/self.mean_dwell_1, self.window, x_fit, 1), 'b', lw=1)           
        ax1.plot(x_fit, icdf(1/self.MLE_dwell_1, self.window, x_fit, 1), 'r', lw=1)     
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Survival probability')
        ax1.set_title('Class 1 (N = %d), True = %.2f [s]' %(len(t1), self.time_b))

        x2, n2 = get_icdf(t2)
        ax2.step(x2, n2, where='mid', c='k', lw=1)
        ax2.plot(x_fit, icdf(1/self.time_b, self.window, x_fit, 2), 'k', lw=1)   
        ax2.plot(x_fit, exp_icdf(1/self.mean_dwell_2, self.window, x_fit, 2), 'b', lw=1)           
        ax2.plot(x_fit, icdf(1/self.MLE_dwell_2, self.window, x_fit, 2), 'r', lw=1)       
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Survival probability')
        ax2.set_title('Class 2 (N = %d), Combined Exp_Finite = %.2f +/- %.2f [s]' %(len(t2), self.MLE_dwell, self.Error_dwell))

        x3, n3 = get_icdf(t3)
        ax3.step(x3, n3, where='mid', c='k', lw=1)  
        ax3.plot(x_fit, icdf(1/self.time_b, self.window, x_fit, 3), 'k', lw=1)   
        ax3.plot(x_fit, exp_icdf(1/self.mean_dwell_3, self.window, x_fit, 3), 'b', lw=1)           
        ax3.plot(x_fit, icdf(1/self.MLE_dwell_3, self.window, x_fit, 3), 'r', lw=1)      
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Survival probability')
        ax3.set_title('Class 3 (N = %d), True (K), Exp (B), Exp_Finite (R)' %(len(t3)))

        ax4.step(x1, n1, where='mid', c='k', lw=1)  
        ax4.plot(x_fit, icdf(1/self.time_b, self.window, x_fit, 1), 'k', lw=1)   
        ax4.plot(x_fit, exp_icdf(1/self.mean_dwell_1, self.window, x_fit, 1), 'b', lw=1)           
        ax4.plot(x_fit, icdf(1/self.MLE_dwell_1, self.window, x_fit, 1), 'r', lw=1)       
        ax4.set_yscale('log')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Survival probability')
        ax4.set_title('Exp = %.2f [s], Exp_Finite = %.2f +/- %.2f [s]' %(self.mean_dwell_1, self.MLE_dwell_1, self.Error_dwell_1))   

        ax5.step(x2, n2, where='mid', c='k', lw=1)  
        ax5.plot(x_fit, icdf(1/self.time_b, self.window, x_fit, 2), 'k', lw=1)   
        ax5.plot(x_fit, exp_icdf(1/self.mean_dwell_2, self.window, x_fit, 2), 'b', lw=1)           
        ax5.plot(x_fit, icdf(1/self.MLE_dwell_2, self.window, x_fit, 2), 'r', lw=1)          
        ax5.set_yscale('log')
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('Survival probability')
        ax5.set_title('Exp = %.2f [s], Exp_Finite = %.2f +/- %.2f [s]' %(self.mean_dwell_2, self.MLE_dwell_2, self.Error_dwell_2)) 

        ax6.step(x3, n3, where='mid', c='k', lw=1)   
        ax6.plot(x_fit, icdf(1/self.time_b, self.window, x_fit, 3), 'k', lw=1)   
        ax6.plot(x_fit, exp_icdf(1/self.mean_dwell_3, self.window, x_fit, 3), 'b', lw=1)           
        ax6.plot(x_fit, icdf(1/self.MLE_dwell_3, self.window, x_fit, 3), 'r', lw=1)      
        ax6.set_yscale('log')
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel('Survival probability')
        ax6.set_title('Exp = %.2f [s], Exp_Finite = %.2f +/- %.2f [s]' %(self.mean_dwell_3, self.MLE_dwell_3, self.Error_dwell_3))   

        fig.savefig(self.dir/'plot8_dwell_icdf.png')   
        plt.close(fig)


    def plot9_wait_icdf(self):

        t1 = self.wait_1
        t2 = self.wait_2
        t3 = self.wait_3
        t_max = max(t1.tolist()+t2.tolist()+t3.tolist())

        bins = np.arange(0, t_max+1, 1)
        x_fit = np.linspace(0, t_max+1, 100)

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(figsize=(20, 10), ncols=3, nrows=2, dpi=300)   

        x1, n1 = get_icdf(t1)
        ax1.step(x1, n1, where='mid', c='k', lw=1)      
        ax1.plot(x_fit, icdf(1/self.time_u, self.n_frame*self.dt, x_fit, 1), 'k', lw=1)   
        ax1.plot(x_fit, exp_icdf(1/self.mean_wait_1, self.window, x_fit, 1), 'b', lw=1)           
        ax1.plot(x_fit, icdf(1/self.MLE_wait_1, self.n_frame*self.dt, x_fit, 1), 'r', lw=1)       
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Survival probability')
        ax1.set_title('Class 1 (N = %d), True = %.2f [s]' %(len(t1), self.time_u))

        x2, n2 = get_icdf(t2)
        ax2.step(x2, n2, where='mid', c='k', lw=1)
        ax2.plot(x_fit, icdf(1/self.time_u, self.n_frame*self.dt, x_fit, 2), 'k', lw=1)   
        ax2.plot(x_fit, exp_icdf(1/self.mean_wait_2, self.window, x_fit, 2), 'b', lw=1)           
        ax2.plot(x_fit, icdf(1/self.MLE_wait_2, self.n_frame*self.dt, x_fit, 2), 'r', lw=1)    
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Survival probability')
        ax2.set_title('Class 2 (N = %d), Combined Exp_Finite = %.2f +/- %.2f [s]' %(len(t2), self.MLE_wait, self.Error_wait))

        x3, n3 = get_icdf(t3)
        ax3.step(x3, n3, where='mid', c='k', lw=1)
        ax3.plot(x_fit, icdf(1/self.time_u, self.n_frame*self.dt, x_fit, 3), 'k', lw=1)   
        ax3.plot(x_fit, exp_icdf(1/self.mean_wait_3, self.window, x_fit, 3), 'b', lw=1)           
        ax3.plot(x_fit, icdf(1/self.MLE_wait_3, self.n_frame*self.dt, x_fit, 3), 'r', lw=1)     
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Survival probability')
        ax3.set_title('Class 3 (N = %d), True (K), Exp (B), Exp_Finite (R)' %(len(t3)))

        ax4.step(x1, n1, where='mid', c='k', lw=1)
        ax4.plot(x_fit, icdf(1/self.time_u, self.n_frame*self.dt, x_fit, 1), 'k', lw=1)   
        ax4.plot(x_fit, exp_icdf(1/self.mean_wait_1, self.window, x_fit, 1), 'b', lw=1)           
        ax4.plot(x_fit, icdf(1/self.MLE_wait_1, self.n_frame*self.dt, x_fit, 1), 'r', lw=1)      
        ax4.set_yscale('log')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Survival probability')
        ax4.set_title('Exp = %.2f [s], Exp_Finite = %.2f +/- %.2f [s]' %(self.mean_wait_1, self.MLE_wait_1, self.Error_wait_1))  

        ax5.step(x2, n2, where='mid', c='k', lw=1)
        ax5.plot(x_fit, icdf(1/self.time_u, self.n_frame*self.dt, x_fit, 2), 'k', lw=1)   
        ax5.plot(x_fit, exp_icdf(1/self.mean_wait_2, self.window, x_fit, 2), 'b', lw=1)           
        ax5.plot(x_fit, icdf(1/self.MLE_wait_2, self.n_frame*self.dt, x_fit, 2), 'r', lw=1)     
        ax5.set_yscale('log')
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('Survival probability')
        ax5.set_title('Exp = %.2f [s], Exp_Finite = %.2f +/- %.2f [s]' %(self.mean_wait_2, self.MLE_wait_2, self.Error_wait_2)) 

        ax6.step(x3, n3, where='mid', c='k', lw=1) 
        ax6.plot(x_fit, icdf(1/self.time_u, self.n_frame*self.dt, x_fit, 3), 'k', lw=1)   
        ax6.plot(x_fit, exp_icdf(1/self.mean_wait_3, self.window, x_fit, 3), 'b', lw=1)           
        ax6.plot(x_fit, icdf(1/self.MLE_wait_3, self.n_frame*self.dt, x_fit, 3), 'r', lw=1)      
        ax6.set_yscale('log')
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel('Survival probability')
        ax6.set_title('Exp = %.2f [s], Exp_Finite = %.2f +/- %.2f [s]' %(self.mean_wait_3, self.MLE_wait_3, self.Error_wait_3))  

        fig.savefig(self.dir/'plot9_wait_icdf.png')   
        plt.close(fig)



    def plot_trace_fit(self):
        # Make a new Trace folder                                                                                                                                                                                                                                                                                         
        trace_dir = self.dir/'Traces'
        if os.path.exists(trace_dir): # Delete if already existing 
            shutil.rmtree(trace_dir)
        os.makedirs(trace_dir)
                
        # Save each trace
        frame = np.arange(self.n_frame)
        n_fig = min(self.save_trace, len(self.trace))        
        for i in range(n_fig):    
            fig, (ax1, ax2) = plt.subplots(figsize=(20, 10), ncols=1, nrows=2, dpi=300)   

            ax1.plot(frame, self.trace[i], 'k', lw=2)
            color = ['b', 'r']
            ax1.plot(frame, self.trace_fit[i], color=color[int(self.is_trace_inlier[i])], lw=2)    
            ax1.axhline(y=self.I_u_inlier.mean(), c='k', ls='--', lw=1) 
            ax1.axhline(y=self.I_b_inlier.mean(), c='k', ls='--', lw=1)     
            ax1.set_ylim([0, 1.5*self.I_b_inlier.mean()])                        
            ax1.set_ylabel('Intensity')
            ax1.set_xlabel('Time [s]')
            title_sp = 'Data (K), Fit: ' + 'Inlier (R)' if self.is_trace_inlier[i] == True else 'Outlier (B)'
            ax1.set_title(title_sp)

            ax2.plot(frame, self.trace[i]-self.trace_fit[i], 'k', lw=2)        
            ax2.axhline(y=0, c='k', ls='-', lw=1)      
            ax2.axhline(y=max(self.rmsd_inlier), c='k', ls='--', lw=1)                     
            ax2.axhline(y=-max(self.rmsd_inlier), c='k', ls='--', lw=1)    
            ax2.set_ylim([-3*max(self.rmsd_inlier), 3*max(self.rmsd_inlier)])            
            ax2.set_ylabel('Intensity')
            ax2.set_xlabel('Time [s]')
            ax2.set_title('Residual')

            fig.subplots_adjust(wspace=0.3, hspace=0.5)
            print("Save Trace %d (%d %%)" % (i+1, ((i+1)/n_fig)*100))
            fig_name = 'Trace%d.png' %(i+1)
            fig.savefig(trace_dir/fig_name) 
            fig.clf()
            plt.close(fig)   

                    
def main():

    print(directory)
    # Find all the movies (*.tif) in the directory tree
    movie_paths = [fn for fn in directory.glob('**/info.txt')]

    # Make a movie instance
    movie = Movie(movie_paths[0])

    # Read info.txt
    movie.read_info()

    mean_dwell_1 = np.zeros((len(n_frames),n_data))
    mean_dwell_2 = np.zeros((len(n_frames),n_data))
    mean_dwell_3 = np.zeros((len(n_frames),n_data))
    MLE_dwell_1 = np.zeros((len(n_frames),n_data))
    MLE_dwell_2 = np.zeros((len(n_frames),n_data))
    MLE_dwell_3 = np.zeros((len(n_frames),n_data))

    mean_wait_1 = np.zeros((len(n_frames),n_data))
    mean_wait_2 = np.zeros((len(n_frames),n_data))
    mean_wait_3 = np.zeros((len(n_frames),n_data))
    MLE_wait_1 = np.zeros((len(n_frames),n_data))
    MLE_wait_2 = np.zeros((len(n_frames),n_data))
    MLE_wait_3 = np.zeros((len(n_frames),n_data))

    # Simulate per n_frame
    for i, n_frame in enumerate(n_frames):
        print(n_frame)

        # Simulate for n_data
        for j in range(n_data):

            # Find peaks where molecules bind
            movie.simulate_peak(n_frame, n_peak)

            # find spots with good signal
            movie.find_spot()

            # Fit spots
            movie.fit_spot()

            # Find binding, unbinding events
            movie.find_event()

            # Exclude short events
            movie.exclude_short()

            # Estimate dwell time
            movie.estimate_time()

            # Save the result into result.txt
            movie.save_result()

            # Plot the result
            print("\nPlotting figures...")  
            movie.plot0_clean()  
            movie.plot5_spot_fit()
            movie.plot6_dwell_pdf()
            movie.plot7_wait_pdf()
            movie.plot8_dwell_icdf()
            movie.plot9_wait_icdf()

            # Plot individual traces
#            movie.plot_trace_fit() 

            # Calculate errorr in percent 
            mean_dwell_1[i][j] = (movie.mean_dwell_1-movie.time_b)/movie.time_b*100
            mean_dwell_2[i][j] = (movie.mean_dwell_2-movie.time_b)/movie.time_b*100
            mean_dwell_3[i][j] = (movie.mean_dwell_3-movie.time_b)/movie.time_b*100

            MLE_dwell_1[i][j] = (movie.MLE_dwell_1-movie.time_b)/movie.time_b*100
            MLE_dwell_2[i][j] = (movie.MLE_dwell_2-movie.time_b)/movie.time_b*100
            MLE_dwell_3[i][j] = (movie.MLE_dwell_3-movie.time_b)/movie.time_b*100            

            mean_wait_1[i][j] = (movie.mean_wait_1-movie.time_u)/movie.time_u*100
            mean_wait_2[i][j] = (movie.mean_wait_2-movie.time_u)/movie.time_u*100
            mean_wait_3[i][j] = (movie.mean_wait_3-movie.time_u)/movie.time_u*100

            MLE_wait_1[i][j] = (movie.MLE_wait_1-movie.time_u)/movie.time_u*100
            MLE_wait_2[i][j] = (movie.MLE_wait_2-movie.time_u)/movie.time_u*100
            MLE_wait_3[i][j] = (movie.MLE_wait_3-movie.time_u)/movie.time_u*100   


    # Plot dwell 
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(figsize=(20, 10), ncols=3, nrows=2, dpi=300)       
    ax1.errorbar(x=n_frames, y=np.mean(mean_dwell_1, axis=1), yerr=np.std(mean_dwell_1, axis=1), marker='o')    
    ax1.axhline(y=0, c='k', ls='-', lw=1)   
    ax1.set_ylabel('Error [%]')
    ax1.set_title('Mean estimator (Class 1)')

    ax2.errorbar(x=n_frames, y=np.mean(mean_dwell_2, axis=1), yerr=np.std(mean_dwell_2, axis=1), marker='o')    
    ax2.axhline(y=0, c='k', ls='-', lw=1)  
    ax2.set_ylabel('Error [%]')
    ax2.set_title('Mean estimator (Class 2)')

    ax3.errorbar(x=n_frames, y=np.mean(mean_dwell_3, axis=1), yerr=np.std(mean_dwell_3, axis=1), marker='o')    
    ax3.axhline(y=0, c='k', ls='-', lw=1)   
    ax3.set_ylabel('Error [%]')
    ax3.set_title('Mean estimator (Class 3)')

    ax4.errorbar(x=n_frames, y=np.mean(MLE_dwell_1, axis=1), yerr=np.std(MLE_dwell_1, axis=1), marker='o')    
    ax4.axhline(y=0, c='k', ls='-', lw=1)  
    ax4.set_xlabel('Number of frame')
    ax4.set_ylabel('Error [%]')
    ax4.set_title('MLE estimator (Class 1)')

    ax5.errorbar(x=n_frames, y=np.mean(MLE_dwell_2, axis=1), yerr=np.std(MLE_dwell_2, axis=1), marker='o')    
    ax5.axhline(y=0, c='k', ls='-', lw=1)   
    ax5.set_xlabel('Number of frame')
    ax5.set_ylabel('Error [%]')
    ax5.set_title('MLE estimator (Class 2)')

    ax6.errorbar(x=n_frames, y=np.mean(MLE_dwell_3, axis=1), yerr=np.std(MLE_dwell_3, axis=1), marker='o')    
    ax6.axhline(y=0, c='k', ls='-', lw=1)    
    ax6.set_xlabel('Number of frame')
    ax6.set_ylabel('Error [%]')
    ax6.set_title('MLE estimator (Class 3)')

    fig.savefig(movie.dir/'plot_dwell.png')   
    plt.close(fig)        

    # Plot wait
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(figsize=(20, 10), ncols=3, nrows=2, dpi=300)       
    ax1.errorbar(x=n_frames, y=np.mean(mean_wait_1, axis=1), yerr=np.std(mean_wait_1, axis=1), marker='o')    
    ax1.axhline(y=0, c='k', ls='-', lw=1)   
    ax1.set_ylabel('Error [%]')
    ax1.set_title('Mean estimator (Class 1)')

    ax2.errorbar(x=n_frames, y=np.mean(mean_wait_2, axis=1), yerr=np.std(mean_wait_2, axis=1), marker='o')    
    ax2.axhline(y=0, c='k', ls='-', lw=1)  
    ax2.set_ylabel('Error [%]')
    ax2.set_title('Mean estimator (Class 2)')

    ax3.errorbar(x=n_frames, y=np.mean(mean_wait_3, axis=1), yerr=np.std(mean_wait_3, axis=1), marker='o')    
    ax3.axhline(y=0, c='k', ls='-', lw=1)   
    ax3.set_ylabel('Error [%]')
    ax3.set_title('Mean estimator (Class 3)')

    ax4.errorbar(x=n_frames, y=np.mean(MLE_wait_1, axis=1), yerr=np.std(MLE_wait_1, axis=1), marker='o')    
    ax4.axhline(y=0, c='k', ls='-', lw=1)  
    ax4.set_xlabel('Number of frame')
    ax4.set_ylabel('Error [%]')
    ax4.set_title('MLE estimator (Class 1)')

    ax5.errorbar(x=n_frames, y=np.mean(MLE_wait_2, axis=1), yerr=np.std(MLE_wait_2, axis=1), marker='o')    
    ax5.axhline(y=0, c='k', ls='-', lw=1)   
    ax5.set_xlabel('Number of frame')
    ax5.set_ylabel('Error [%]')
    ax5.set_title('MLE estimator (Class 2)')

    ax6.errorbar(x=n_frames, y=np.mean(MLE_wait_3, axis=1), yerr=np.std(MLE_wait_3, axis=1), marker='o')    
    ax6.axhline(y=0, c='k', ls='-', lw=1)    
    ax6.set_xlabel('Number of frame')
    ax6.set_ylabel('Error [%]')
    ax6.set_title('MLE estimator (Class 3)')
    
    fig.savefig(movie.dir/'plot_wait.png')   
    plt.close(fig)   


if __name__ == "__main__":
    main()


