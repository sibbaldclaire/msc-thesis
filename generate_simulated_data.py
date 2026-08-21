#Import key packages
import numpy as np
import numpy.random as random
from astropy.timeseries import LombScargle
from scipy import stats
from astropy.table import Table, vstack
from scipy.optimize import root_scalar
import scipy.special as sp
import sys
from pathlib import Path
sys.path.append(str(Path("~/msc-thesis/python_utils").expanduser()))
import observation_time as ot
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
import h5py

# Set the number of each type of dataset
N_BKG = 10
N_SINGLE = 20
N_DOUBLE = 20

# Create 8 background only simulations
sampler = LatinHypercube(d=2, seed=4)
sample = sampler.random(n=N_BKG)

SNR = 0.2+sample[:,0]*(5-0.2)
sigma2 = (1/SNR)**2
#sigma2 = 1+sample[:,0]*(10-1)
period = 0.2+sample[:,1]*(1000-0.2)
omega_w1 = 1/period * 2*np.pi
omega_w2 = np.repeat(0, N_BKG)
beta_a1 = np.repeat(0, N_BKG)
beta_b1 = np.repeat(0, N_BKG)
beta_a2 = np.repeat(0, N_BKG)
beta_b2 = np.repeat(0, N_BKG)

tab_bkg = Table([beta_a1, beta_b1, beta_a2, beta_b2, omega_w1, omega_w2, sigma2], names=['a1', 'b1', 'a2', 'b2', 'w1', 'w2', 'sigma2'])

# Create 20 single sinusoid simulations
sampler = LatinHypercube(d=4, seed=81)
sample = sampler.random(n=N_SINGLE)

SNR = 0.2+sample[:,0]*(5-0.2)
sigma2 = (1/SNR)**2
#sigma2 = 1+sample[:,0]*(10-1)
period = 0.2+sample[:,1]*(1000-0.2)
omega_w1 = 1/period * 2*np.pi
beta_a1 = sample[:,2]*(5)
beta_b1 = sample[:,3]*(5)
beta_a2 = np.repeat(0, N_SINGLE)
beta_b2 = np.repeat(0, N_SINGLE)
omega_w2 = np.repeat(0, N_SINGLE)

tab_single = Table([beta_a1, beta_b1, beta_a2, beta_b2, omega_w1, omega_w2, sigma2], names=['a1', 'b1', 'a2', 'b2', 'w1', 'w2', 'sigma2'])

# Create 20 double sinusoid simulations
sampler = LatinHypercube(d=7, seed=87)
sample = sampler.random(n=N_DOUBLE)

SNR = 0.2+sample[:,0]*(5-0.2)
sigma2 = (1/SNR)**2
#sigma2 = 1+sample[:,0]*(10-1)
period1 = 0.2+sample[:,1]*(1000-0.2)
omega_w1 = 1/period1 * 2*np.pi
period2 = 0.2+sample[:,2]*(1000-0.2)
omega_w2 = 1/period2 * 2*np.pi
beta_a1 = sample[:,3]*(5)
beta_b1 = sample[:,4]*(5)
beta_a2 = sample[:,5]*(5)
beta_b2 = sample[:,6]*(5)

tab_double = Table([beta_a1, beta_b1, beta_a2, beta_b2, omega_w1, omega_w2, sigma2], names=['a1', 'b1', 'a2', 'b2', 'w1', 'w2', 'sigma2'])

tab = vstack([tab_bkg, tab_single, tab_double])
tab['t'] = np.empty(len(tab), dtype=object)
tab['y'] = np.empty(len(tab), dtype=object)

# Generate simulated datasets
for ind, row in enumerate(tab):
    if row['w2']==0:
        period = 1/(row['w1']/2/np.pi)
    else:
        period = np.max([1/(row['w1']/2/np.pi), 1/(row['w2']/2/np.pi)])
    t = ot.generate_time(period)
    N = len(t)
    beta = np.array([row['a1'], row['b1'], row['a2'], row['b2']])
    Xw = np.column_stack([np.cos(row['w1']*t), np.sin(row['w1']*t), np.cos(row['w2']*t), np.sin(row['w2']*t)])
    y = Xw @ beta + random.normal(0, np.sqrt(row['sigma2']), N)

    tab['t'][ind] = [t]
    tab['y'][ind] = [y]

with h5py.File("~/simulation_datasets/Simulation_Datasets.h5", "w") as f:

    for i, row in enumerate(tab):
        grp = f.create_group(f"lightcurve_{i}")

        grp["t"] = row["t"]
        grp["y"] = row["y"]

        grp.attrs["a1"] = row["a1"]
        grp.attrs["b1"] = row["b1"]
        grp.attrs["a2"] = row["a2"]
        grp.attrs["b2"] = row["b2"]
        grp.attrs["w1"] = row["w1"]
        grp.attrs["w2"] = row["w2"]
        grp.attrs["sigma2"] = row["sigma2"]

    
    




