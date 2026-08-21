#Import key packages
from scipy.special import logsumexp
import numpy as np
import numpy.random as random
import sys
from pathlib import Path
sys.path.append(str(Path("~/python_utils").expanduser()))
import lombscargle_tools as ls
from astropy.timeseries import LombScargle
from scipy.optimize import minimize
from scipy import stats
import scipy.special as sp

# Key functions
def inverse_gamma_logpdf(x, a, b):
    '''
    Compute the inverse gamma log pdf.
    '''
    return a*np.log(b) - np.log(sp.gamma(a)) - (1+a)*np.log(x) - b/x

def inverse_covariance(sigma2, tau2, X):
    '''
    Uses the Woodbury Identity to compute the inverse of the covariance matrix.
    '''
    Xt = X.T
    In = np.identity(np.shape(X)[0])
    Ip = np.identity(np.shape(X)[1])

    return 1/sigma2*In - tau2/sigma2**2 * X @ np.linalg.inv(Ip + tau2/sigma2*Xt@X) @ Xt

def log_determinant(sigma2, tau2, X):
    '''
    Uses the matrix determinant lemma to compute the log determinant
    '''
    n = np.shape(X)[0]
    p = np.shape(X)[1]
    Xt = X.T
    Ip = np.identity(p)

    return n*np.log(sigma2) + np.linalg.slogdet(Ip + tau2 * Xt @ X / sigma2)[1]

def log_marginal_bkg(y, a, b):
    N = len(y)
    return sp.loggamma((2*a+N)/2)-sp.loggamma(a)-N/2*np.log(2*a*np.pi*b)+N/2*np.log(a)-(2*a+N)/2*np.log(1+1/(2*b)*y.T @ y)

def log_marginal_like(y, Xw, sigma2, tau2):
    n = len(y)
    p = Xw.shape[1]
    Xt = Xw.T
    Ip = np.identity(p)
    XtX = Xt @ Xw
    A = Ip + (tau2/sigma2)*XtX          # p x p
    A_inv = np.linalg.inv(A)             # p x p

    log_det = n*np.log(sigma2) + np.linalg.slogdet(A)[1]

    Xty = Xt @ y                         # p
    quad = (1/sigma2)*np.dot(y, y) - (tau2/sigma2**2) * (Xty @ A_inv @ Xty)

    c = (-n/2)*np.log(2*np.pi) - 0.5*log_det
    exp = -0.5*quad

    return c+exp

def find_peak_single(t,y):
    freq, power = LombScargle(t, y).autopower()
    return freq[np.where(power==np.max(power))][0]*2*np.pi

def find_peak_double(t,y):
    # Coarse-to-fine search (see lombscargle_tools.find_double_peak) instead
    # of a dense K=500 x K=500 grid -- ~20x fewer evaluations for comparable
    # or better resolution. Already returns angular frequencies.
    return ls.find_double_peak(t, y)

def laplace_evidence_1d(t, y, tau2, lamda, w0, sigma20, a, b):
    """
    Laplace approximation over (w, log(sigma2)).
    """
    # Initial guess
    theta0 = np.array([w0, np.log(sigma20)])

    def g_single(theta):
        w, eta = theta
        sigma2 = np.exp(eta)
        return (log_integrand(w, sigma2, tau2, lamda, t, y, a, b) + eta)

    # Minimise negative log target
    res = minimize(lambda th: -g_single(th), x0=theta0, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-12})
    theta_hat = res.x

    # Numerical Hessian
    h = 1e-6
    H = np.zeros((2, 2))

    for i in range(2):
        for j in range(2):
            ei = np.zeros(2)
            ej = np.zeros(2)
            ei[i] = h
            ej[j] = h
            H[i, j] = (g_single(theta_hat+ei+ej)-g_single(theta_hat+ei-ej)-g_single(theta_hat-ei+ej)+g_single(theta_hat-ei-ej))/(4*h**2)
    sign, logdet = np.linalg.slogdet(-H)
    if sign <= 0:
        raise RuntimeError("Hessian is not negative definite at the mode.")

    return g_single(theta_hat) + np.log(2*np.pi) - 0.5*logdet

def laplace_evidence_2d(t, y, tau2, lamda, w0, sigma20, a, b):
    """
    Laplace approximation over (w1, w2, log(sigma2))
    """
    #Initial guess
    theta0 = np.array([w0[0], w0[1], np.log(sigma20)])

    def g_double(theta):
        w1, w2, eta = theta
        sigma2 = np.exp(eta)
        return (log_integrand_double(w1, w2, sigma2, tau2, lamda, t, y, a, b) + eta)

    #Minimise negative lof target
    res = minimize(lambda th: -g_double(th), x0=theta0, method="Nelder-Mead", options={"xatol":1e-10, "fatol":1e-12})
    theta_hat = res.x

    #Numerical Hessian
    h = 1e-6
    H = np.zeros((3,3))

    for i in range(3):
        for j in range(3):
            ei = np.zeros(3)
            ej = np.zeros(3)
            ei[i] = h
            ej[j] = h

            H[i,j] = (g_double(theta_hat+ei+ej)-g_double(theta_hat+ei-ej)-g_double(theta_hat-ei+ej)+g_double(theta_hat-ei-ej))/(4*h*h)

    sign, logdet = np.linalg.slogdet(-H)
    if sign <= 0:
        raise RuntimeError("Hessian is not negative definite.")

    return (g_double(theta_hat) + 1.5*np.log(2*np.pi) - 0.5*logdet)


# Single
def compute_BF_single(t, y, tau2=1, lamda=1, a=1, b=0.5, w0_single=None):
    #Single
    if w0_single is None:
        w0_single = find_peak_single(t,y)
    X0_single = np.column_stack([np.cos(w0_single*t), np.sin(w0_single*t)])
    beta0_single, *_ = np.linalg.lstsq(X0_single, y, rcond=None)
    sigma20_single = np.var(y, mean=X0_single @ beta0_single)
    log_p_signal = laplace_evidence_1d(t, y, tau2=tau2, lamda=lamda, w0=w0_single, sigma20=sigma20_single, a=a, b=b)

    #Background
    N = len(t)
    In = np.identity(N)
    log_p_bkg = stats.multivariate_t.logpdf(y, shape=b/a*In, df=2*a)

    log_BF = log_p_signal-log_p_bkg

    return log_BF

def log_integrand(w, sigma2, tau2, lamda, t, y, a, b):
    Xw = np.column_stack([np.cos(w*t), np.sin(w*t)])

    return -lamda*w+np.log(lamda) + log_marginal_like(y, Xw, sigma2, tau2) + inverse_gamma_logpdf(sigma2, a, b)

# Double
def log_integrand_double(w1, w2, sigma2, tau2, lamda, t, y, a, b):
    Xw = np.column_stack([np.cos(w1*t), np.sin(w1*t), np.cos(w2*t), np.sin(w2*t)])

    return -lamda*w1+np.log(lamda) - lamda*w2+np.log(lamda) + log_marginal_like(y, Xw, sigma2, tau2) + inverse_gamma_logpdf(sigma2, a, b)

def compute_BF_double(t, y, tau2=1, lamda=1, a=1, b=0.5, w0_single=None, w0_double=None):
    #Single
    if w0_single is None:
        w0_single = find_peak_single(t,y)
    X0_single = np.column_stack([np.cos(w0_single*t), np.sin(w0_single*t)])
    beta0_single, *_ = np.linalg.lstsq(X0_single, y, rcond=None)
    sigma20_single = np.var(y, mean=X0_single @ beta0_single)
    log_p_single = laplace_evidence_1d(t, y, tau2=tau2, lamda=lamda, w0=w0_single, sigma20=sigma20_single, a=a, b=b)

    #Double
    if w0_double is None:
        w1_0, w2_0 = find_peak_double(t, y)
    else:
        w1_0, w2_0 = w0_double
    X0_double = np.column_stack([np.cos(w1_0*t), np.sin(w1_0*t), np.cos(w2_0*t), np.sin(w2_0*t)])
    beta0_double, *_ = np.linalg.lstsq(X0_double, y, rcond=None)
    sigma20_double = np.var(y, mean=X0_double @ beta0_double)
    log_p_double = laplace_evidence_2d(t, y, tau2=tau2, lamda=lamda, w0=[w1_0, w2_0], sigma20=sigma20_double, a=a, b=b)
    
    log_BF = log_p_double - log_p_single
    return log_BF
