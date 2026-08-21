#Import key packages
import numpy as np
import numpy.random as random
from astropy.timeseries import LombScargle
from scipy import stats
from astropy.table import Table
from scipy.optimize import root_scalar
from matplotlib.ticker import MultipleLocator
import argparse, os, glob
from multiprocessing import Pool, cpu_count
import scipy.special as sp
import sys
from pathlib import Path
sys.path.append(str(Path("~/python_utils").expanduser()))
import lombscargle_tools as ls

# Define required functions
def count_upcrossings(threshold, data):
    '''
    Counts the number of times data crosses threshold
    '''
    data = np.asarray(data)
    return np.sum((data[1:] > threshold) & (data[:-1] < threshold))
    
# Single   
def compute_profile_statistic_single(y, mu):
    N = len(y)
    return N*(np.log(np.sum(y**2)) - np.log(np.sum((y-mu)**2)))#1/sigma2*np.sum(2*y*mu-mu**2)

def MC_upcrossings_single(t, y, nrep=10000, threshold=0.5):
    upcrossings = np.zeros(nrep)
    statistic = np.zeros(nrep)
    omega_max = np.zeros(nrep)
    angulars = np.linspace(1/np.max(t)*2*np.pi, 1/np.min(t)*2*np.pi, 2000)
    WT = np.outer(angulars, t)
    COS = np.cos(WT)
    SIN = np.sin(WT)
    Xs = np.stack([COS, SIN], axis=-1)
    
    XtX_inv = np.linalg.inv(np.einsum('kni,knj->kij', Xs, Xs))
    sigma = np.std(y, mean=0)

    for i in range(nrep):
        #print(f'Complete: {round(i/nrep*100, 2)}%', end='\r')
        stat, up = MC_iteration_single(t, sigma, Xs, XtX_inv, COS, SIN, threshold)
        omega_max[i] = angulars[np.where(stat == np.max(stat))][0]
        statistic[i] = np.max(stat)
        upcrossings[i] = up

    return omega_max, statistic, np.mean(upcrossings)

def MC_iteration_single(t, sigma, Xs, XtX_inv, COS, SIN, threshold):
    #Generate simulated data
    N = len(t)
    y = random.normal(0, sigma, N)

    Xty = np.einsum('kni,n->ki', Xs, y)      # (k, p)
    betas = np.einsum('kij,kj->ki', XtX_inv, Xty)

    mu = betas[:, 0:1] * COS + betas[:, 1:2] * SIN
    profile_statistic = N*(np.log(np.sum(y**2)) - np.log(np.sum((y-mu)**2, axis=1)))
    return profile_statistic, count_upcrossings(threshold, profile_statistic)

def compute_test_statistic_single(t, y):   
    #Compute test statistic
    angulars = np.linspace(1/np.max(t)*2*np.pi, 1/np.min(t)*2*np.pi, 1000)
    profile_statistic = []
        
    for w in angulars:
        Xest = np.column_stack([np.cos(w*t), np.sin(w*t)])
        beta, *_ = np.linalg.lstsq(Xest, y, rcond=None)
        aest = beta[0]
        best = beta[1]
        mu = aest*np.cos(w*t)+best*np.sin(w*t)
        
        profile_statistic.append(compute_profile_statistic_single(y, mu))

    
    c = np.max(profile_statistic) # This is the test statistic
    return c
    
def likelihood_ratio_test_single(t, y, s=2, extremal=False):
    #Estimate number of upcrossings under null
    s = 2
    c0 = s-1
    
    w_hat, qw, Nc0 = MC_upcrossings_single(t, y, threshold=c0)
    c = compute_test_statistic_single(t, y)
    
    Nc = Nc0*(c/c0)**((s-1)/2)*np.exp(-(c-c0)/2)
    prob = 1-stats.chi2.cdf(c, s) + Nc

    if extremal:
        return c, prob, qw
    else:
        return c, prob

#Double
def compute_profile_statistic_double(y, mu01, mu1, mu2):
    N = len(y)
    return N*(np.log(np.sum((y-mu01)**2)) - np.log(np.sum((y-mu1-mu2)**2)))

def MC_upcrossings_double(t, y, w1, beta1, nrep=10000, threshold=0.5):
    upcrossings = np.zeros(nrep)
    statistic = np.zeros(nrep)
    omega_max = np.zeros(nrep)
    
    ang = np.repeat(w1, len(t))
    angulars = np.linspace(1/np.max(t)*2*np.pi, 1/np.min(t)*2*np.pi, len(t))

    W1T = np.outer(ang, t)
    COS1 = np.cos(W1T)
    SIN1 = np.sin(W1T)
    W2T = np.outer(angulars, t)
    COS2 = np.cos(W2T)
    SIN2 = np.sin(W2T)
    
    Xs = np.stack([COS1, SIN1, COS2, SIN2], axis=-1)

    XtX_inv = np.linalg.inv(np.einsum('kni,knj->kij', Xs, Xs))
    mu1 = beta1[0] * COS1 + beta1[1] * SIN1
    sigma = np.std(y, mean=mu1)

    for i in range(nrep):
        #print(f'Complete: {round(i/nrep*100, 2)}%', end='\r')
        stat, up = MC_iteration_double(t, sigma, mu1, Xs, XtX_inv, COS1, SIN1, COS2, SIN2, threshold)
        omega_max[i] = angulars[np.where(stat == np.max(stat))][0]
        statistic[i] = np.max(stat)
        upcrossings[i] = up

    return omega_max, statistic, np.mean(upcrossings)

def MC_iteration_double(t, sigma, mu1, Xs, XtX_inv, COS1, SIN1, COS2, SIN2, threshold):
    #Generate simulated data
    N = len(t)
    y = random.normal(mu1[0], sigma, N)

    Xty = np.einsum('kni,n->ki', Xs, y)      # (k, p)
    betas = np.einsum('kij,kj->ki', XtX_inv, Xty)

    mu1_refit = betas[:, 0:1] * COS1 + betas[:, 1:2] * SIN1
    mu2_refit = betas[:, 2:3] * COS2 + betas[:, 3:4] * SIN2
    profile_statistic = N*(np.log(np.sum((y-mu1)**2, axis=1)) - np.log(np.sum((y-mu1_refit-mu2_refit)**2, axis=1)))
    return profile_statistic, count_upcrossings(threshold, profile_statistic)

def compute_test_statistic_double(t, y, w1_est, beta1_est):   
    #Compute test statistic
    angulars = np.linspace(1/np.max(t)*2*np.pi, 1/np.min(t)*2*np.pi, 1000)
    profile_statistic = []
        
    for w in angulars:
        Xest = np.stack([np.cos(w1_est*t), np.sin(w1_est*t), np.cos(w*t), np.sin(w*t)], axis=-1)

        beta2_est, *_ = np.linalg.lstsq(Xest, y, rcond=None)
    
        mu2 = beta2_est[2]*np.cos(w*t)+beta2_est[3]*np.sin(w*t)
        mu1 = beta2_est[0]*np.cos(w1_est*t)+ beta2_est[1]*np.sin(w1_est*t)
        mu01 = beta1_est[0]*np.cos(w1_est*t)+ beta1_est[1]*np.sin(w1_est*t)
        
        profile_statistic.append(compute_profile_statistic_double(y, mu01, mu1, mu2))
    
    c = np.max(profile_statistic) # This is the test statistic

    return c

def likelihood_ratio_test_double(t, y, s=2, extremal=False, w1_est=None):
    #Estimate number of upcrossings under null
    if w1_est is None:
        frequency, power = LombScargle(t, y).autopower()
        w1_est = frequency[np.where(power==np.max(power))]*2*np.pi

    w1_est = ls.refine_single_frequency(t, y, w1_est)
    X_est = np.stack([np.cos(w1_est*t), np.sin(w1_est*t)], axis=-1)
    beta1_est, *_ = np.linalg.lstsq(X_est, y, rcond=None)
    
    s = 2
    c0 = s-1
    
    w_hat, qw, Nc0 = MC_upcrossings_double(t, y, w1_est, beta1_est, threshold=c0)
    c = compute_test_statistic_double(t, y, w1_est, beta1_est)
    
    Nc = Nc0*(c/c0)**((s-1)/2)*np.exp(-(c-c0)/2)
    prob = 1-stats.chi2.cdf(c, s) + Nc

    if extremal:
        return c, prob, qw
    else:
        return c, prob
    
    


