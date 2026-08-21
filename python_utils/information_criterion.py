import numpy as np
from astropy.timeseries import LombScargle
import sys
from pathlib import Path
sys.path.append(str(Path("~/python_utils").expanduser()))
import lombscargle_tools as ls

def background_model(t, y):
    return 0, 1

def single_model(t, y, w_hat=None):
    #Estimate w if not passed
    if w_hat is None:
        frequency, power = LombScargle(t, y).autopower()
        w_hat = frequency[np.where(power == np.max(power))]*2*np.pi

    #Estimate beta
    X = np.stack([np.cos(w_hat*t), np.sin(w_hat*t)], axis=-1)
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)

    return beta_hat[0]*np.cos(w_hat*t)+beta_hat[1]*np.sin(w_hat*t), 4

def double_model(t, y, w1_hat=None, w2_hat=None):
    #Estimate w1, w2 if not passed
    if w1_hat is None or w2_hat is None:
        w1_hat, w2_hat = ls.find_double_peak(t, y)

    #Estimate beta
    X = np.stack([np.cos(w1_hat*t), np.sin(w1_hat*t), np.cos(w2_hat*t), np.sin(w2_hat*t)], axis=-1)
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)

    return beta_hat[0]*np.cos(w1_hat*t)+beta_hat[1]*np.sin(w1_hat*t) + beta_hat[2]*np.cos(w2_hat*t)+beta_hat[3]*np.sin(w2_hat*t), 7

def loglikelihood_normal(x, mean=0, var=1):
    N = len(x)
    return -N/2*np.log(2*np.pi) - N/2*np.log(var) - 1/(2*var)*np.sum(x-mean)
    
def AIC(t, y, model, **model_kwargs):
    mu_hat, k = model(t, y, **model_kwargs)
    sigma2_hat = np.var(y, mean=mu_hat)
    return -2*loglikelihood_normal(y, mu_hat, sigma2_hat)+2*k

def BIC(t, y, model, **model_kwargs):
    mu_hat, k = model(t, y, **model_kwargs)
    sigma2_hat = np.var(y, mean=mu_hat)
    return -2*loglikelihood_normal(y, mu_hat, sigma2_hat)+k*np.log(len(y))