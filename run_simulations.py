import os
# Cap BLAS thread count per worker process. Must be set before numpy (or
# anything that imports numpy) is imported -- otherwise each of the
# n_workers processes also tries to multithread its own linear algebra
# calls, and they end up fighting each other for cores.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path('~/python_utils').expanduser()))
import lombscargle_tools as ls
import likelihood_ratio_test as lrt
import bayes_factor as bf
import information_criterion as ic
import h5py
from astropy.timeseries import LombScargle
from concurrent.futures import ProcessPoolExecutor
import scipy.special as sp
import pandas as pd

H5_PATH = 'Simulation_Datasets.h5'

# One open HDF5 file handle per worker process, instead of opening and
# closing the file for every single lightcurve.
_h5file = None

def _init_worker(path):
    global _h5file
    _h5file = h5py.File(path, 'r')

# Define functions
def estimate_extremal_index(qw, s=2):
    n = len(qw)
    gamma = np.euler_gamma
    beta = np.sqrt(6)*np.std(qw)/np.pi #Method of moments to get these
    mu = np.mean(qw)-beta*gamma #Method of moments to get these
    dn = 2*(np.log(n)+(s/2-1)*np.log(np.log(n))-np.log(sp.gamma(s/2)))
    theta_MLE = np.exp(-0.5*(dn+beta*np.log(1/n*np.sum(np.exp(-qw/beta)))))
    
    return theta_MLE

def process_lightcurve(index):
    print(f'Completing: lightcurve {index}', end='\r')
    grp = _h5file[f'lightcurve_{index}']
    t = grp['t'][:][0]
    y = grp['y'][:][0]

    # Estimate the single- and double-sinusoid frequency peaks ONCE per
    # lightcurve, then reuse them in the LRT, Bayes factor, and information
    # criterion steps below (each of which previously re-estimated its own
    # peak from scratch, including two separate expensive double-frequency
    # grid searches).
    frequency, power = LombScargle(t, y).autopower()
    w_single = frequency[np.argmax(power)]*2*np.pi
    w1_double, w2_double = ls.find_double_peak(t, y)

    #Likelihood ratio test
    c_single, pval_single, qw = lrt.likelihood_ratio_test_single(t, y, s=2, extremal=True)
    theta_s = estimate_extremal_index(qw, s=2)
    c_double, pval_double, qw = lrt.likelihood_ratio_test_double(t, y, s=2, extremal=True, w1_est=w_single)
    theta_d = estimate_extremal_index(qw, s=2)
    
    #Bayes Factor
    bf_single = bf.compute_BF_single(t, y, tau2=1, lamda=1, a=1, b=0.5, w0_single=w_single)
    bf_double = bf.compute_BF_double(t, y, tau2=1, lamda=1, a=1, b=0.5,
                                      w0_single=w_single, w0_double=(w1_double, w2_double))
    
    #Information Criterion
    aic_bkg = ic.AIC(t, y, ic.background_model)
    aic_single = ic.AIC(t, y, ic.single_model, w_hat=w_single)
    aic_double = ic.AIC(t, y, ic.double_model, w1_hat=w1_double, w2_hat=w2_double)

    bic_bkg = ic.BIC(t, y, ic.background_model)
    bic_single = ic.BIC(t, y, ic.single_model, w_hat=w_single)
    bic_double = ic.BIC(t, y, ic.double_model, w1_hat=w1_double, w2_hat=w2_double)
    
    return {'index': index,'AIC_bkg': aic_bkg, 'AIC_single': aic_single, 'AIC_double': aic_double, 
        'BIC_bkg': bic_bkg, 'BIC_single': bic_single, 'BIC_double': bic_double,
        'LRT_single': pval_single, 'LRT_double': pval_double, 
        'theta_MLE_s': theta_s, 'theta_MLE_d': theta_d,
        'BF_single': bf_single, 'BF_double': bf_double}
    
def main():
    n_lightcurves = 3
    n_workers = 5
    path = '~/msc-thesis/simulated_data/simulation_results_data.csv'

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                              initargs=(H5_PATH,)) as pool:
        results = list(pool.map(process_lightcurve, range(n_lightcurves)))

    df = pd.DataFrame(results)
    df.to_csv(path)
    print(f"Results saved to {path}")

if __name__ == "__main__":
    main()