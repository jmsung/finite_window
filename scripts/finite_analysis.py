"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Created by Jongmin Sung (jongmin.sung@gmail.com)

Single molecule binding and unbinding analysis for anaphase promoting complex (apc) 

class Data() 
- path, name, load(), list[], n_movie, movies = [Movie()], plot(), analysis(), spot_size, frame_rate, 
- path, name, n_row, n_col, n_frame, pixel_size, 
- background, spots = [], molecules = [Molecule()]

class Molecule()
- position (row, col), intensity 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from __future__ import division, print_function, absolute_import
import numpy as np
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

directory = data_dir
#directory = data_dir/'19-05-29 Movies 300pix300pi'

# ---------------------------------------------------------------------------

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

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

def get_mean_time(t, t_max):
    mean_time = np.mean(t)
    for _ in range(100):
        mean_time = np.mean(t) + t_max/(np.exp(t_max/mean_time)-1)
    return mean_time

def prob_1(k, T, t):
    t = np.array(t)
    return k*np.exp(-k*t)/(1-np.exp(-k*T))

def prob_2(k, T, t):
    t = np.array(t)
    return (k*T-k*t)/(k*T-1+np.exp(-k*T))*k*np.exp(-k*t)

def LL_1(param, T, t):      
    [k] = param
    return np.sum(np.log10(prob_1(k, T, t)))  

def LL_2(param, T, t):      
    [k] = param
    return np.sum(np.log10(prob_2(k, T, t)))  

def MLE_1(k, T, t): 
    fun = lambda *args: -LL_1(*args)
    p0 = [k]
    result = minimize(fun, p0, method='SLSQP', args=(T, t)) 
    return 1/result["x"][0]

def MLE_2(k, T, t): 
    fun = lambda *args: -LL_2(*args)
    p0 = [k]
    result = minimize(fun, p0, method='SLSQP', args=(T, t)) 
    return 1/result["x"][0]


class Movie:
    def __init__(self, path):
        self.path = path
        self.dir = path.parent
        self.name = path.name

    # Read info.txt and movie.tif   
    def read_movie(self):  
        # Read info.txt
        self.info = {}
        with open(Path(self.dir/'info.txt')) as f:
            for line in f:
                line = line.replace(" ", "") # remove white space
                if line == '\n': # skip empty line
                    continue
                (key, value) = line.rstrip().split("=")
                self.info[key] = value
        self.dt = float(self.info['time_interval'])
        self.spot_size = int(self.info['spot_size'])
        self.save_trace = int(self.info['save_trace'])
        
        # Read movie.tif
        with TiffFile(self.path) as tif:
            imagej_hyperstack = tif.asarray()
            imagej_metadata = str(tif.imagej_metadata)
            self.metadata = imagej_metadata.split(',')

        # write meta_data    
        with open(self.path.parent/'meta_data.txt', 'w') as f:
            for item in imagej_metadata:
                f.write(item+'\n')

        # Crop the image to make the size integer multiple of 10
        self.bin_size = 20
        self.n_frame = np.size(imagej_hyperstack, 0)
        n_row = np.size(imagej_hyperstack, 1)
        self.n_row = int(int(n_row/self.bin_size)*self.bin_size)        
        n_col = np.size(imagej_hyperstack, 2) 
        self.n_col = int(int(n_col/self.bin_size)*self.bin_size)
        self.I_original = imagej_hyperstack[:,:self.n_row,:self.n_col]

        print('[frame, row, col] = [%d, %d, %d]' %(self.n_frame, self.n_row, self.n_col))  


    def correct_offset(self):
#        self.I_original_min = np.median(np.min(self.I_original, axis=0))
#        self.I_offset = self.I_original - self.I_original_min
        self.I_offset = self.I_original.copy()


    def correct_flatfield(self):
        self.I_offset_max = np.max(self.I_offset, axis=0)
        self.I_flatfield = self.I_offset.copy()

        # Flatfield correct
        if str2bool(self.info['flatfield_correct']) == True:
            print('flatfield_correct = True')

            # Masking from local threshold        
            self.mask = self.I_offset_max > threshold_local(self.I_offset_max, block_size=51, offset=-31) 
            self.I_mask = self.I_offset_max*self.mask 
            self.I_mask_out = self.I_offset_max*(1-self.mask) 

            # Local averaging signals
            self.I_bin = np.zeros((self.n_row, self.n_col))
            m = self.bin_size
            for i in range(int(self.n_row/m)):
                for j in range(int(self.n_col/m)):
                    window = self.I_mask[i*m:(i+1)*m, j*m:(j+1)*m].flatten()          
                    signals = [signal for signal in window if signal > 0]
                    if signals:
                        self.I_bin[i*m:(i+1)*m,j*m:(j+1)*m] = np.mean(signals)

            # Fill empty signal with the local mean. 
            for i in range(int(self.n_row/m)):
                for j in range(int(self.n_col/m)):
                    if self.I_bin[i*m,j*m] == 0:
                        window = self.I_bin[max(0,(i-1)*m):min((i+2)*m,self.n_row), max(0,(j-1)*m):min((j+2)*m,self.n_col)].flatten() 
                        signals = [signal for signal in window if signal > 0] # Take only positive signal
                        if signals:
                            self.I_bin[i*m:(i+1)*m,j*m:(j+1)*m] = np.mean(signals)   

            # Remaining empty signal will be filled with the global mean. 
            self.I_bin[self.I_bin==0] = np.mean(self.I_bin[self.I_bin>0])

            # Smoothening the sharp bolder.
            self.I_bin_filter = gaussian_filter(self.I_bin, sigma=10)

            # Flatfield correct by normalization
            self.I_flatfield = np.array(self.I_offset)
            for i in range(self.n_frame):
                self.I_flatfield[i,] = self.I_offset[i,] / self.I_bin_filter * np.max(self.I_bin_filter)    

            # Local averaging signals after flatfield correction
            self.I_flatfield_max = np.max(self.I_flatfield, axis=0)
            self.I_flatfield_mask = self.I_flatfield_max*self.mask 
            self.I_flatfield_bin = np.zeros((self.n_row, self.n_col))
            for i in range(int(self.n_row/m)):
                for j in range(int(self.n_col/m)):
                    window = self.I_flatfield_mask[i*m:(i+1)*m, j*m:(j+1)*m].flatten()          
                    signals = [signal for signal in window if signal > 0]
                    if signals:
                        self.I_flatfield_bin[i*m:(i+1)*m,j*m:(j+1)*m] = np.mean(signals)
        else:
            print('flatfield_correct = False')


    def correct_drift(self):
        self.I_drift = self.I_flatfield.copy()

        # Drift correct
        if str2bool(self.info['drift_correct']) == True:
            print('drift_correct = True')

            I = self.I_flatfield.copy()
            I_ref = I[int(len(I)/2),] # Mid frame as a reference frame

            # Translation as compared with I_ref
            d_row = np.zeros(len(I), dtype='int')
            d_col = np.zeros(len(I), dtype='int')
            for i, I_frame in enumerate(I):
                result = translation(I_ref, I_frame)
                d_row[i] = round(result['tvec'][0])
                d_col[i] = round(result['tvec'][1])      

            # Changes of translation between the consecutive frames
            dd_row = d_row[1:] - d_row[:-1]
            dd_col = d_col[1:] - d_col[:-1]

            # Sudden jump in translation set to zero
            step_limit = 2
            dd_row[abs(dd_row)>step_limit] = 0
            dd_col[abs(dd_col)>step_limit] = 0

            # Adjusted translation
            d_row[0] = 0
            d_col[0] = 0
            d_row[1:] = np.cumsum(dd_row)
            d_col[1:] = np.cumsum(dd_col)

            # Offset mid to zero
            self.drift_row = d_row 
            self.drift_col = d_col      

            # Running avg
            self.drift_row = running_avg(self.drift_row, 5)
            self.drift_col = running_avg(self.drift_col, 5)      

            # Offset to zero
            self.drift_row = self.drift_row - self.drift_row[0]  
            self.drift_col = self.drift_col - self.drift_col[0]  

            # Translate images
            for i in range(len(I)):
                self.I_drift[i,] = np.roll(self.I_drift[i,], self.drift_row[i], axis=0)
                self.I_drift[i,] = np.roll(self.I_drift[i,], self.drift_col[i], axis=1)        
        else:
            print('drift_correct = False')
      
        # Simple name after the corrections
        self.I = self.I_drift.copy()
        self.I_max = np.max(self.I, axis=0)


    # Find spots where molecules bind
    def find_peak(self):
        # Find local maxima from I_max
        self.peak = peak_local_max(self.I_max, min_distance=int(self.spot_size*1.0))        
        self.n_peak = len(self.peak[:, 1])
        self.peak_row = self.peak[::-1,0]
        self.peak_col = self.peak[::-1,1]

        # Get the time trace of each spots
        self.peak_trace = np.zeros((self.n_peak, self.n_frame))
        for i in range(self.n_peak):
            # Get the trace from each spot
            r = self.peak_row[i]
            c = self.peak_col[i]
            s = int((self.spot_size-1)/2)
            self.peak_trace[i] = np.sum(np.sum(self.I[:,r-s:r+s+1,c-s:c+s+1], axis=2), axis=1)/self.spot_size**2


    # Evaluate quality of signal
    def find_spot(self):
        # Find inliers with I_max and I_min
        self.peak_min = np.min(self.peak_trace, axis=1)
        self.peak_max = np.max(self.peak_trace, axis=1)
        self.is_peak_min_inlier = is_inlier(self.peak_min, 3)
        self.is_peak_max_inlier = is_inlier(self.peak_max, 3)
        self.is_peak_inlier = self.is_peak_min_inlier & self.is_peak_max_inlier

        self.n_spot = sum(self.is_peak_inlier)
        self.trace = self.peak_trace[self.is_peak_inlier]
        self.spot_row = self.peak_row[self.is_peak_inlier]        
        self.spot_col = self.peak_col[self.is_peak_inlier]   

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
        self.is_rmsd_inlier = self.rmsd < np.median(self.rmsd)*2
        self.is_I_u_inlier = is_inlier(self.I_u, 3)
        self.is_I_b_inlier = is_inlier(self.I_b, 3)
        self.is_trace_inlier = self.is_rmsd_inlier & self.is_I_u_inlier & self.is_I_b_inlier

        # Save inlier traces
        self.state_inlier = self.state[self.is_trace_inlier]
        self.trace_inlier = self.trace[self.is_trace_inlier]
        self.I_u_inlier = self.I_u[self.is_trace_inlier]        
        self.I_b_inlier = self.I_b[self.is_trace_inlier]
        self.rmsd_inlier = self.rmsd[self.is_trace_inlier]

        print('Found', self.n_peak, 'peaks. ')     
        print('Rejected', self.n_peak - len(self.rmsd_inlier), 'outliers.')   



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
                dt_odd = dt[::2]
                dt_even = dt[1:2]

                if state[0] == 0: # Odd events are d2, event events are w2
                    self.dwell_2.extend(dt_odd)
                    self.wait_2.extend(dt_even)
                else: # Odd events are w2, event events are d2
                    self.wait_2.extend(dt_odd)
                    self.dwell_2.extend(dt_even)                

    def  exclude_short(self):
        self.dwell_1 = [t for t in self.dwell_1 if t > 1]
        self.dwell_2 = [t for t in self.dwell_2 if t > 1]
        self.dwell_3 = [t for t in self.dwell_3 if t > 1]
        self.wait_1 = [t for t in self.wait_1 if t > 1]
        self.wait_2 = [t for t in self.wait_2 if t > 1]
        self.wait_3 = [t for t in self.wait_3 if t > 1]


    def estimate_time(self):

        self.mean_dwell_1 = get_mean_time(self.dwell_1, self.n_frame)
        self.mean_dwell_2 = get_mean_time(self.dwell_2, self.n_frame)
        self.mean_dwell_3 = get_mean_time(self.dwell_3, self.n_frame)
        self.mean_wait_1 = get_mean_time(self.wait_1, self.n_frame)
        self.mean_wait_2 = get_mean_time(self.wait_2, self.n_frame)
        self.mean_wait_3 = get_mean_time(self.wait_3, self.n_frame)

        # MLE estimation of dwell time
        self.MLE_dwell_1 = MLE_1(0.5/np.mean(self.dwell_1), self.n_frame, self.dwell_1)
        self.MLE_dwell_2 = MLE_2(0.5/np.mean(self.dwell_2), self.n_frame, self.dwell_2)
        self.MLE_dwell_3 = MLE_1(0.5/np.mean(self.dwell_3), self.n_frame, self.dwell_3)
        self.MLE_wait_1 = MLE_1(0.5/np.mean(self.wait_1), self.n_frame, self.wait_1)
        self.MLE_wait_2 = MLE_2(0.5/np.mean(self.wait_2), self.n_frame, self.wait_2)
        self.MLE_wait_3 = MLE_1(0.5/np.mean(self.wait_3), self.n_frame, self.wait_3)                                


    def save_result(self):
        pass        
      

    def plot0_clean(self):
        # clean all existing png files in the folder
        files = os.listdir(self.dir)    
        for file in files:
            if file.endswith('png'):
                os.remove(self.dir/file)    


    def plot1_original_min_max(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(20, 10), ncols=2, nrows=2, dpi=300)

        I_min = np.min(self.I_original, axis=0)
        I_max = np.max(self.I_original, axis=0)

        sp = ax1.imshow(I_min, cmap='gray')
        fig.colorbar(sp, ax=ax1) 

        ax2.hist(I_min.ravel(), 20, histtype='step', lw=2, color='k')    
        ax2.set_yscale('log')
        ax2.set_xlim(0, np.max(I_max)) 
        ax2.set_xlabel('Intensity')
        ax2.set_ylabel('Counts')
        ax2.set_title('Min projection - original')

        sp = ax3.imshow(I_max, cmap='gray')
        fig.colorbar(sp, ax=ax3) 

        ax4.hist(I_max.ravel(), 50, histtype='step', lw=2, color='k')                      
        ax4.set_yscale('log')
        ax4.set_xlim(0, np.max(I_max)) 
        ax4.set_xlabel('Intensity')
        ax4.set_ylabel('Counts')
        ax4.set_title('Max projection - original')

        fig.tight_layout()
        fig.savefig(self.dir/'plot1_original_min_max.png')   
        plt.close(fig)                                                                                                                                                                                                                                                                                                                                                                                                                                                            


    def plot2_flatfield(self):              
        if str2bool(self.info['flatfield_correct']) == False:
            return None

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(figsize=(20, 10), ncols=3, nrows=2, dpi=300)

        sp = ax1.imshow(self.I_offset_max, cmap=cm.gray)
        fig.colorbar(sp, ax=ax1) 
        ax1.set_title('Max intensity - original')      
  
        ax2.imshow(self.mask, cmap=cm.gray)
        ax2.set_title('Mask')           

        sp = ax3.imshow(self.I_bin, cmap=cm.gray)
        fig.colorbar(sp, ax=ax3) 
        ax3.set_title('Intensity - bin')

        sp = ax4.imshow(self.I_bin_filter, cmap=cm.gray)
        fig.colorbar(sp, ax=ax4) 
        ax4.set_title('Intensity - bin filter')        

        sp = ax5.imshow(self.I_max, cmap=cm.gray)
        fig.colorbar(sp, ax=ax5) 
        ax5.set_title('Max intensity - flatfield')

        sp = ax6.imshow(self.I_flatfield_bin, cmap=cm.gray)
        fig.colorbar(sp, ax=ax6) 
        ax6.set_title('Intensity flatfield - bin')

        fig.tight_layout()
        fig.savefig(self.dir/'plot2_flatfield.png')   
        plt.close(fig)


    def plot3_drift(self):                      
        I_row = np.squeeze(self.I[:,int(self.n_row/2),:])
        I_col = np.squeeze(self.I[:,:,int(self.n_row/2)])

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(figsize=(20, 10), ncols=2, nrows=3, dpi=300)

        ax1.plot(self.drift_row, 'k')
        ax1.set_yticks(np.arange(min(self.drift_row), max(self.drift_row)+1, 1.0))
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Pixel')
        ax1.set_title('Drift in row')

        ax2.plot(self.drift_col, 'k')
        ax2.set_yticks(np.arange(min(self.drift_col), max(self.drift_col)+1, 1.0))
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Pixel')
        ax2.set_title('Drift in column')

        ax3.imshow(I_col, cmap='gray')
        ax3.set_xlabel('Column')
        ax3.set_ylabel('Frame')

        ax4.imshow(I_row, cmap='gray')
        ax4.set_xlabel('Row')
        ax4.set_ylabel('Frame')

        ax5.plot(np.mean(I_col, axis=0), 'ko-')
        ax5.set_xlim([0, self.n_col])
        ax5.set_xlabel('Column')

        ax6.plot(np.mean(I_row, axis=0), 'ko-')
        ax6.set_xlim([0, self.n_row])
        ax6.set_xlabel('Row')

        fig.tight_layout()
        fig.savefig(self.dir/'plot3_drift.png')   
        plt.close(fig)


    def plot4_spot(self):
        fig = plt.figure(figsize=(20, 10), dpi=300)
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[:, 0])
        ax1.imshow(self.I_max, cmap=cm.gray)
        color = [['b','r'][int(i)] for i in self.is_peak_inlier] 
        ax1.scatter(self.peak_col, self.peak_row, lw=0.8, s=50, facecolors='none', edgecolors=color)
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Row')        
        ax1.set_title('Spots: selected (R), rejected (B)')  

        bins = np.linspace(min(self.peak_min), max(self.peak_min), 50)     
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.peak_min, bins = bins, histtype='step', lw=2, color='b')
        ax2.hist(self.peak_min[self.is_peak_min_inlier], bins = bins, histtype='step', lw=2, color='r')   
        ax2.set_xlabel('Intensity')
        ax2.set_ylabel('Counts')     
        ax2.set_title('Intensity min')

        bins = np.linspace(min(self.peak_max), max(self.peak_max), 50)     
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(self.peak_max, bins = bins, histtype='step', lw=2, color='b')
        ax3.hist(self.peak_max[self.is_peak_max_inlier], bins = bins, histtype='step', lw=2, color='r')
        ax3.set_xlabel('Intensity')
        ax3.set_ylabel('Counts')    
        ax3.set_title('Intensity max')

        fig.savefig(self.dir/'plot4_spot.png')   
        plt.close(fig)


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


    def plot6_dwell_time(self):

        t1 = self.dwell_1
        t2 = self.dwell_2
        t3 = self.dwell_3
        t_max = max(t1+t2+t3)

        n_bin = 15

        if n_bin < t_max:
            interval = np.ceil(t_max/n_bin)
            bins = np.arange(0.5, interval*n_bin+0.5, interval)
        else:
            interval = 1
            bins = np.arange(0.5, t_max+0.5, interval)  

        x_fit = np.linspace(0.5, t_max, 100)

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(figsize=(20, 10), ncols=3, nrows=2, dpi=300)
  
        ax1.hist(t1, bins=bins, histtype='step', lw=2, color='k')
        ax1.plot(x_fit, len(t1)/(1-np.exp(-self.n_frame/self.mean_dwell_1))*interval*np.exp(-x_fit/self.mean_dwell_1)/self.mean_dwell_1, 'r')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Counts')
        ax1.set_title('Dwell time (class 1): '+"{0:.2f}".format(np.mean(self.dwell_1))+' (mean), '+"{0:.2f}".format(self.MLE_dwell_1)+' (MLE)'+' [frames]')

        ax2.hist(t2, bins=bins, histtype='step', lw=2, color='k')
        ax2.plot(x_fit, len(t2)/(1-np.exp(-self.n_frame/self.mean_dwell_2))*interval*np.exp(-x_fit/self.mean_dwell_2)/self.mean_dwell_2, 'r')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Counts')
        ax2.set_title('Dwell time (class 2): '+"{0:.2f}".format(np.mean(self.dwell_2))+' (mean), '+"{0:.2f}".format(self.MLE_dwell_2)+' (MLE)'+' [frames]')
  
        ax3.hist(t3, bins=bins, histtype='step', lw=2, color='k')
        ax3.plot(x_fit, len(t3)/(1-np.exp(-self.n_frame/self.mean_dwell_3))*interval*np.exp(-x_fit/self.mean_dwell_3)/self.mean_dwell_3, 'r')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Counts')
        ax3.set_title('Dwell time (class 3): '+"{0:.2f}".format(np.mean(self.dwell_3))+' (mean), '+"{0:.2f}".format(self.MLE_dwell_3)+' (MLE)'+' [frames]')
  
        ax4.hist(t1, bins=bins, histtype='step', lw=2, color='k')
        ax4.plot(x_fit, len(t1)/(1-np.exp(-self.n_frame/self.mean_dwell_1))*interval*np.exp(-x_fit/self.mean_dwell_1)/self.mean_dwell_1, 'r')
        ax4.set_yscale('log')
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Log(Counts)')
  
        ax5.hist(t2, bins=bins, histtype='step', lw=2, color='k')
        ax5.plot(x_fit, len(t2)/(1-np.exp(-self.n_frame/self.mean_dwell_2))*interval*np.exp(-x_fit/self.mean_dwell_2)/self.mean_dwell_2, 'r')
        ax5.set_yscale('log')
        ax5.set_xlabel('Frame')
        ax5.set_ylabel('Log(Counts)')
   
        ax6.hist(t3, bins=bins, histtype='step', lw=2, color='k')
        ax6.plot(x_fit, len(t3)/(1-np.exp(-self.n_frame/self.mean_dwell_3))*interval*np.exp(-x_fit/self.mean_dwell_3)/self.mean_dwell_3, 'r')
        ax6.set_yscale('log')
        ax6.set_xlabel('Frame')
        ax6.set_ylabel('Log(Counts)')

        fig.savefig(self.dir/'plot6_dwell_time.png')   
        plt.close(fig)


    def plot7_wait_time(self):

        t1 = self.wait_1
        t2 = self.wait_2
        t3 = self.wait_3
        t_max = max(t1+t2+t3)

        n_bin = 15

        if n_bin < t_max:
            interval = np.ceil(t_max/n_bin)
            bins = np.arange(0.5, interval*n_bin+0.5, interval)
        else:
            bins = np.arange(0.5, t_max+0.5, 1)

        x_fit = np.linspace(0.5, t_max, 100)

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(figsize=(20, 10), ncols=3, nrows=2, dpi=300)   
 
        ax1.hist(t1, bins=bins, histtype='step', lw=2, color='k')
        ax1.plot(x_fit, len(t1)/(1-np.exp(-self.n_frame/self.mean_wait_1))*interval*np.exp(-x_fit/self.mean_wait_1)/self.mean_wait_1, 'r')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Counts')
        ax1.set_title('Wait time (class 1): '+"{0:.2f}".format(np.mean(self.wait_1))+' (mean), '+"{0:.2f}".format(self.MLE_wait_1)+' (MLE)'+' [frames]')
 
        ax2.hist(t2, bins=bins, histtype='step', lw=2, color='k')
        ax2.plot(x_fit, len(t2)/(1-np.exp(-self.n_frame/self.mean_wait_2))*interval*np.exp(-x_fit/self.mean_wait_2)/self.mean_wait_2, 'r')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Counts')
        ax2.set_title('Wait time (class 1): '+"{0:.2f}".format(np.mean(self.wait_2))+' (mean), '+"{0:.2f}".format(self.MLE_wait_2)+' (MLE)'+' [frames]')
 
        ax3.hist(t3, bins=bins, histtype='step', lw=2, color='k')
        ax3.plot(x_fit, len(t3)/(1-np.exp(-self.n_frame/self.mean_wait_3))*interval*np.exp(-x_fit/self.mean_wait_3)/self.mean_wait_3, 'r')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Counts')
        ax3.set_title('Wait time (class 1): '+"{0:.2f}".format(np.mean(self.wait_3))+' (mean), '+"{0:.2f}".format(self.MLE_wait_3)+' (MLE)'+' [frames]')

        ax4.hist(t1, bins=bins, histtype='step', lw=2, color='k')
        ax4.plot(x_fit, len(t1)/(1-np.exp(-self.n_frame/self.mean_wait_1))*interval*np.exp(-x_fit/self.mean_wait_1)/self.mean_wait_1, 'r')
        ax4.set_yscale('log')
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Log(Counts)')

        ax5.hist(t2, bins=bins, histtype='step', lw=2, color='k')
        ax5.plot(x_fit, len(t2)/(1-np.exp(-self.n_frame/self.mean_wait_2))*interval*np.exp(-x_fit/self.mean_wait_2)/self.mean_wait_2, 'r')
        ax5.set_yscale('log')
        ax5.set_xlabel('Frame')
        ax5.set_ylabel('Log(Counts)')
  
        ax6.hist(t3, bins=bins, histtype='step', lw=2, color='k')
        ax6.plot(x_fit, len(t3)/(1-np.exp(-self.n_frame/self.mean_wait_3))*interval*np.exp(-x_fit/self.mean_wait_3)/self.mean_wait_3, 'r')
        ax6.set_yscale('log')
        ax6.set_xlabel('Frame')
        ax6.set_ylabel('Log(Counts)')

        fig.savefig(self.dir/'plot7_wait_time.png')   
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
            ax1.set_xlabel('Frame')
            title_sp = 'Data (K), Fit: ' + 'Inlier (R)' if self.is_trace_inlier[i] == True else 'Outlier (B)'
            ax1.set_title(title_sp)

            ax2.plot(frame, self.trace[i]-self.trace_fit[i], 'k', lw=2)        
            ax2.axhline(y=0, c='k', ls='-', lw=1)      
            ax2.axhline(y=max(self.rmsd_inlier), c='k', ls='--', lw=1)                     
            ax2.axhline(y=-max(self.rmsd_inlier), c='k', ls='--', lw=1)    
            ax2.set_ylim([-3*max(self.rmsd_inlier), 3*max(self.rmsd_inlier)])            
            ax2.set_ylabel('Intensity')
            ax2.set_xlabel('Frame')
            ax2.set_title('Residual')

            fig.subplots_adjust(wspace=0.3, hspace=0.5)
            print("Save Trace %d (%d %%)" % (i+1, ((i+1)/n_fig)*100))
            fig_name = 'Trace%d.png' %(i+1)
            fig.savefig(trace_dir/fig_name) 
            fig.clf()
            plt.close(fig)   

                    
def main():
    # Find all the movies (*.tif) in the directory tree
    movie_paths = [fn for fn in directory.glob('**/*.tif')]
#                   if not fn.name == 'GFP.tif']

    print('%d movies are found' %(len(movie_paths)))

    # Run through each movie
    for i, movie_path in enumerate(movie_paths):
        print('='*100)
        print('Movie #%d/%d' %(i+1, len(movie_paths)))
        print('Path:', movie_path.parent)
        print('Name:', movie_path.name)

        # Check info.txt exist.
        info_file = Path(movie_path.parent/'info.txt')
        if not info_file.exists():
            print('info.txt does not exist.')
            continue

        # Make a movie instance
        movie = Movie(movie_path)

        # Read the movie
        movie.read_movie()

        # Corrections: offset, flatfield, drift
        movie.correct_offset()
        movie.correct_flatfield()        
        movie.correct_drift()

        # Find peaks where molecules bind
        movie.find_peak()

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
        movie.plot1_original_min_max()     
        movie.plot2_flatfield()              
        movie.plot3_drift()             
        movie.plot4_spot()    
        movie.plot5_spot_fit()
        movie.plot6_dwell_time()
        movie.plot7_wait_time()
        movie.plot_trace_fit() 


if __name__ == "__main__":
    main()



"""
To-do

* cutoff t=1 and offset=1
* MLE error? > weighted mean
* plot three: sample mean, mean with cutoff, MLE 


* Simulation: our vs old methods 
* Henrik's method for accurate time. 

* simplify the fitting function 
* exclude dwell time = 1  
* Weighted mean 
* flatfield: subtract bg (use same algorithm with bg)
* animated movie in each trace (or cross-section)
* try 5 spot size?
* peak finding algorithm. use random walk, add noise and sampling after steps
or gradient decendent

# If the seperation of two peaks are less than the emission noise, then something is wrong. It's pick the noise. 

* User defined thresholding 

* Better drift correction algorithm? use cross-section?
* better fit for doulbe gaussian? polynomial?
* set dwell_max, dwell_min
* Save results in a text (panda)
* Seperate code to read text and combine or compare




"""
