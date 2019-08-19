# SiMPull simulation

from __future__ import division, print_function, absolute_import
import numpy as np
from numpy.random import rand, randn
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import shutil

np.random.seed()

n_mol = 500
n_frame = 200
n_delay = 50
n_corr = n_frame-1
corr = np.arange(1, n_corr)

t_b = 5
t_u = 50
t_t = 1/(1/t_b+1/t_u)
noise = 0.2 # Noise to signal
blink = 0 # percent per frame
nonspecific = 0 # percent per frame

SNR_min = 5
dwell_min = 3
wait_min = 3
noise_cutoff = 0.6
tpf = 0.05

def reject_outliers(data, m = 3.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def running_avg(x, n):
    return np.convolve(x, np.ones((n,))/n, mode='valid')
   
# Exponential function with cutoff at x = b 
def Exp_cutoff(a, b, x):
    return np.exp(-(x-b)/a)/a * (0.5*(np.sign(x-b)+1))

def Exp(a, x):
    return np.exp(-(x)/a)/a * (0.5*(np.sign(x)+1))
     
def exp1(x, a, b, c):
    return a + b * np.exp(-x/c) 

def exp2(x, a, b, c, d, e):
    return a + b * np.exp(-x/c) + d* np.exp(-x/e) 

class Mol(object):
    def __init__(self, I):  
        self.I_frame = I
        
    def normalize(self):
        I = self.I_frame
#        x = running_avg(I, 3)
#        I = np.array([x[0]]+x.tolist()+[x[-1]])   
        
        I = I - np.min(I)
        I = I/np.max(I)

        bg_u = np.mean(I[I < 0.5])
        bg_b = np.mean(I[I > 0.5])
        self.I_frame = (I - bg_u)/(bg_b - bg_u) 
               
    def find_noise(self):
        self.I_avg = running_avg(self.I_frame, 3)
        noise0 = self.I_frame[1:-1]-self.I_avg
        noise1 = reject_outliers(noise0)
        self.noise = np.std(noise1)    

    def evaluate(self, SNR_min, dwell_min):
        blinking = 1

        if self.noise > 1/SNR_min: return False        
        
        x = running_avg(self.I_frame, 3)
        self.I_s = np.array([x[0]]+x.tolist()+[x[-1]])       
#        SNR = self.I_s / self.noise
#        signal = SNR > SNR_min
         
#        self.I_s = self.I_frame
        signal = self.I_s > noise_cutoff

        t_b = []
        t_ub = []
        for i in range(len(signal)-1):
            if (signal[i] == False) & (signal[i+1] == True):
                t_b.append(i)
            if (signal[i] == True) & (signal[i+1] == False):
                t_ub.append(i)
        
        if len(t_b)*len(t_ub) == 0: return False 
        if t_ub[0] < t_b[0]: # remove pre-existing binding
            del t_ub[0]
        if len(t_b)*len(t_ub) == 0: return False                
        if t_ub[-1] < t_b[-1]: # remove unfinished binding
            del t_b[-1]
        if len(t_b)*len(t_ub) == 0: return False      

        # combine blinking
        blink_ub = []
        blink_b = []             
        if len(t_b) > 1:  
            for i in range(len(t_b)-1):   
                if abs(t_ub[i] - t_b[i+1]) <= blinking: 
                    blink_ub.append(t_ub[i])
                    blink_b.append(t_b[i+1])
                  
            if len(blink_ub) > 0:
                for i in range(len(blink_ub)):
                    t_ub.remove(blink_ub[i])
                    t_b.remove(blink_b[i])

        # delete too short or too long binding
        transient_ub = []
        transient_b = []
        for i in range(len(t_b)):                                      
            if (t_ub[i] - t_b[i] < dwell_min): 
                transient_ub.append(t_ub[i])
                transient_b.append(t_b[i])
                
        if len(transient_b) > 0:
            for i in range(len(transient_b)):
                t_ub.remove(transient_ub[i])
                t_b.remove(transient_b[i])

        if len(t_b)*len(t_ub) == 0: return False    
              
        self.dwell = []
        self.wait = []     
        self.SNR = []
        self.I_fit = np.zeros(len(signal))          
        for i in range(len(t_b)): 
            self.dwell.append(t_ub[i] - t_b[i])
            if i > 0:
                self.wait.append(t_b[i] - t_ub[i-1])
            I_mean = np.mean(self.I_frame[t_b[i]+1:t_ub[i]-1])
            self.SNR.append(I_mean/self.noise)            
            self.I_fit[t_b[i]+1:t_ub[i]+1] = I_mean
        return True
 



class Data(object):
    def __init__(self):
        pass
        
    def generate(self):
        I = np.zeros((n_mol, n_frame+n_delay))
        
        for i in range(n_mol):
            for j in range(n_frame+n_delay-1):
                if I[i][j] == 0:
                    if rand() < 1 - np.exp(-1/t_u):
                        I[i][j+1] = 1
                else:
                    if rand() > 1 - np.exp(-1/t_b):
                        I[i][j+1] = 1         
                    if rand() < blink:
                        I[i][j] = 0
                if rand() < nonspecific:
                    I[i][j] = I[i][j] + 1                                                  
            I[i] = I[i] + noise*randn(n_frame+n_delay)            
        self.I = I[:,n_delay:]

    def normalize(self):
        I = self.I
        for i in range(n_mol):
            I[i] = I[i] - np.min(I[i])
            I[i] = I[i]/np.max(I[i])

            bg_u = np.mean(I[i][I[i] < 0.5])
            bg_b = np.mean(I[i][I[i] > 0.5])
            I[i] = (I[i] - bg_u)/(bg_b - bg_u) 
                        

    def analyze(self):       
        self.corr_b = np.zeros((n_mol, n_corr-1))
        for i in range(n_mol):
            for j in corr:
                corr_b = []
                for k in range(n_frame-j):
                    corr_b.append(self.I[i][k]*self.I[i][k+j])
                self.corr_b[i, j-1] = np.mean(corr_b)
        self.corr_b_mean = np.mean(self.corr_b, axis=0)
        self.corr_b_sem = np.std(self.corr_b, axis=0)/n_mol**0.5


    # Find real molecules from the peaks
    def find_mols(self, SNR_min, dwell_min): 
        self.mols = []
        self.dwells = []
        self.waits = []
        self.noise = []
        self.SNR = []
        for i in range(n_mol):
            mol = Mol(self.I[i])
            mol.normalize()
            mol.find_noise()
            if mol.evaluate(SNR_min, dwell_min) is True:
                self.mols.append(mol)    
                self.dwells.extend(mol.dwell)
                self.waits.extend(mol.wait)
                self.noise.append(mol.noise)
                self.SNR.extend(mol.SNR)
        print('Found', len(self.mols), 'molecules. \n')  
        self.dwell_mean = np.mean(self.dwells)-min(self.dwells)
        self.dwell_std = np.std(self.dwells)
        self.dwell_min = np.min(self.dwells)
        self.wait_mean = np.mean(self.waits)-min(self.waits)
        self.wait_std = np.std(self.waits)
        self.wait_min = np.min(self.waits)


    def plot(self):
        # Figure 1
        fig1 = plt.figure(1, figsize = (20, 10), dpi=300)  
        row = 5
        col = 4  
        for i in range(row*col):        
            sp = fig1.add_subplot(row, col, i+1)  
            sp.plot(self.I[i], 'k-')
            sp.axhline(y=0, color='b', linestyle='dashed', linewidth=1)
            sp.axhline(y=1, color='b', linestyle='dashed', linewidth=1)
        fig1.savefig('Fig1.png')
        plt.close('all')
                        
        fig2 = plt.figure(2, figsize = (20, 10), dpi=300)  
        for i in range(row*col):        
            sp = fig2.add_subplot(row, col, i+1)  
            sp.plot(self.corr_b[i], 'k-')    
            sp.axhline(y=0, color='b', linestyle='dashed', linewidth=1) 
        fig2.savefig('Fig2.png')
        plt.close('all')
                        
        # Figure 3. Correlation - Full range
        fig3 = plt.figure(3, figsize = (20, 10), dpi=300)  
        p0 = [0, 1, 10]
        p1, pcov1 = curve_fit(exp1, corr, self.corr_b_mean, p0, self.corr_b_sem) 
        x_fit = np.linspace(0,max(corr),1000)
        y_fit1 = exp1(x_fit, p1[0], p1[1], p1[2])        
        scale1 = y_fit1[0]
        offset1 = p1[0]
        y_fit1 = (y_fit1 - offset1)/(scale1 - offset1)
        self.corr_b_mean1 = (self.corr_b_mean - offset1)/(scale1 - offset1)    

        sp1 = fig3.add_subplot(121)
        sp1.plot(corr, self.corr_b_mean1, 'ko', mfc='none')
        sp1.plot(x_fit, y_fit1, 'r', linewidth=2)     
        sp1.set_xlim([0, max(corr)])  
        sp1.set_ylim([-0.1, 1])     
        title = "[Given] t_b = %.1f, t_u = %.1f, t_tot = %.1f " % (t_b, t_u, t_t)
        sp1.set_title(title)
        sp1.set_xlabel('Lag time [Frame]')
        sp1.set_ylabel('Correlation [AU]')
        
        sp2 = fig3.add_subplot(122)
        sp2.plot(corr, self.corr_b_mean1, 'ko', mfc='none')
        sp2.set_yscale('log')
        sp2.semilogy(x_fit, y_fit1, 'r', linewidth=2)    
        sp2.set_xlim([0, max(corr)])  
        sp2.set_ylim([min(y_fit1)/10, 1])          
        title = "[Estimate] t_est = %.1f +/- %.1f" % (p1[2], pcov1[2,2]**0.5)
        sp2.set_title(title)
        sp2.set_xlabel('Lag time [frame]')
        sp2.set_ylabel('Correlation [AU]')
        
        fig3.savefig('Fig3.png')
        plt.close('all')
  
    
        # Figure 4. Correlation - Short range
        fig4 = plt.figure(4, figsize = (20, 10), dpi=300)  
        p0 = [p1[0], p1[1], p1[2]]
        lim = corr < p1[2]*2
        p1, pcov1 = curve_fit(exp1, corr[lim], self.corr_b_mean[lim], p0, self.corr_b_sem[lim]) 
        x_fit = np.linspace(0,max(corr[lim]),1000)
        y_fit1 = exp1(x_fit, p1[0], p1[1], p1[2])        
        scale1 = y_fit1[0]
        offset1 = p1[0]
        y_fit1 = (y_fit1 - offset1)/(scale1 - offset1)
        self.corr_b_mean1 = (self.corr_b_mean - offset1)/(scale1 - offset1)    

        sp1 = fig4.add_subplot(121)
        sp1.plot(corr[lim], self.corr_b_mean1[lim], 'ko', mfc='none')
        sp1.plot(x_fit, y_fit1, 'r', linewidth=2)     
        sp1.set_xlim([0, max(corr[lim])])  
        sp1.set_ylim([0, 1])     
        title = "[Given] t_b = %.1f, t_u = %.1f, t_tot = %.1f " % (t_b, t_u, t_t)
        sp1.set_title(title)
        sp1.set_xlabel('Lag time [Frame]')
        sp1.set_ylabel('Correlation [AU]')
        
        sp2 = fig4.add_subplot(122)
        sp2.plot(corr[lim], self.corr_b_mean1[lim], 'ko', mfc='none')
        sp2.set_yscale('log')
        sp2.semilogy(x_fit, y_fit1, 'r', linewidth=2)    
        sp2.set_xlim([0, max(corr[lim])])  
        sp2.set_ylim([min(y_fit1)/2, 1])          
        title = "[Estimate] t_est = %.1f +/- %.1f" % (p1[2], pcov1[2,2]**0.5)
        sp2.set_title(title)
        sp2.set_xlabel('Lag time [frame]')
        sp2.set_ylabel('Correlation [AU]')                 

        fig4.savefig('Fig4.png')
        plt.close('all')

        # Figure 5
        fig5 = plt.figure(5, figsize = (20, 10), dpi=300)  
        
        sp1 = fig5.add_subplot(221)                                 
        hist_lifetime = sp1.hist(self.dwells, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
        n_lifetime = len(self.dwells)*(hist_lifetime[1][1] - hist_lifetime[1][0])
        x_lifetime = np.linspace(0, max(self.dwells), 1000)
        y_mean = n_lifetime*Exp_cutoff(self.dwell_mean, self.dwell_min, x_lifetime) 
        sp1.plot(x_lifetime, y_mean, 'r', linewidth=2)  
        title = '\nMean dwell time [s] = %.2f +/- %.2f (N = %d)' % (tpf*self.dwell_mean, tpf*self.dwell_std/(len(self.dwells)**0.5), len(self.dwells))
        sp1.set_title(title)
        print(title)
                    
        sp2 = fig5.add_subplot(222) 
        hist_lifetime = sp2.hist(self.dwells, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
        sp2.set_yscale('log')
        sp2.semilogy(x_lifetime, y_mean, 'r', linewidth=2)
        sp2.axis([0, 1.1*max(x_lifetime), 0.1, 2*max(y_mean)])
        title = 'Mean dwell time [frame] = %.1f +/- %.1f (N = %d)' % (self.dwell_mean, self.dwell_std/(len(self.dwells)**0.5), len(self.dwells))
        sp2.set_title(title)

        sp3 = fig5.add_subplot(223)                                 
        hist_lifetime = sp3.hist(self.waits, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
        n_lifetime = len(self.waits)*(hist_lifetime[1][1] - hist_lifetime[1][0])
        x_lifetime = np.linspace(0, max(self.waits), 1000)
        y_mean = n_lifetime*Exp_cutoff(self.wait_mean, self.wait_min, x_lifetime) 
        sp3.plot(x_lifetime, y_mean, 'r', linewidth=2)  
        title = '\nMean dwell time [s] = %.2f +/- %.2f (N = %d)' % (tpf*self.wait_mean, tpf*self.wait_std/(len(self.dwells)**0.5), len(self.waits))
        sp3.set_title(title)
        print(title)
                    
        sp4 = fig5.add_subplot(224) 
        hist_lifetime = sp4.hist(self.waits, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
        sp4.set_yscale('log')
        sp4.semilogy(x_lifetime, y_mean, 'r', linewidth=2)
        sp4.axis([0, 1.1*max(x_lifetime), 0.1, 2*max(y_mean)])
        title = 'Mean dwell time [frame] = %.1f +/- %.1f (N = %d)' % (self.wait_mean, self.wait_std/(len(self.waits)**0.5), len(self.waits))
        sp4.set_title(title)

        fig5.savefig('Fig5.png')
        plt.close('all')
                                                                                                                                                                                                                                                  
        # Figure for individual traces    
        save_trace = input('Save individual traces [y/n]? ')
        if save_trace == 'y':    
            percent = int(input('How much percent [1-100]? '))    
            if percent < 1:
                percent = 1
            if percent > 100:
                percent = 100
                
            i_fig = 1               
            n_col = 1
            n_row = 1
#            self.data_path = os.getcwd()
#            directory = self.data_path+'\\Figures'
#            if os.path.exists(directory):
#                shutil.rmtree(directory)
#                os.makedirs(directory)
#            else:
#                os.makedirs(directory)
                
            n_fig = int(n_mol*percent/100)
                
            for j in range(n_fig):
                self.frame = np.arange(n_frame)
                k = j%(n_col*n_row)
                if k == 0:                      
                    fig = plt.figure(i_fig+10, figsize = (25, 15), dpi=300)
                    i_fig += 1
                sp = fig.add_subplot(n_row, n_col, k+1)
                sp.plot(self.frame, self.mols[j].I_frame, 'k.', linewidth=1, markersize=3)
                sp.plot(self.frame, self.mols[j].I_s, 'b', linewidth=1, markersize=3)
                sp.plot(self.frame, self.mols[j].I_fit, 'r', linewidth=2, markersize=3)
                sp.axhline(y=0, color='k', linestyle='dashed', linewidth=1)
                sp.axhline(y=noise_cutoff, color='k', linestyle='dotted', linewidth=1)
                sp.axhline(y=1, color='k', linestyle='dotted', linewidth=1)                
                title_sp = '(noise = %.2f)' % (self.mols[j].noise)
                sp.set_title(title_sp)
                fig.subplots_adjust(wspace=0.3, hspace=0.5)
                if (k == n_col*n_row-1) | (j == n_fig-1):
                    print("Save Fig %d (%d %%)" % (i_fig-1, (j/n_mol)*100+1))
                    fig.savefig("Figs%d.png" % (i_fig-1)) 
                    fig.clf()
                    plt.close('all')   
                                               
      
# Start  
plt.close('all')
data = Data()
print('Generating data...')
data.generate()
print('Normalize...')
data.normalize()
print('Binding event detection...')
data.find_mols(SNR_min, dwell_min)
print('Analyzing correlation...')
data.analyze()
print('Plotting...')
data.plot()
