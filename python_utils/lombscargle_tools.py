import numpy as np
from astropy.timeseries import LombScargle
from scipy.optimize import minimize

def single_Lomb_Scargle(t, y, K=12500):
    N = len(t)
    frequencies = np.linspace(0, 1/np.min(t), K)
    C = np.cos(2*np.pi*frequencies[:,None]*t[None:,])
    S = np.sin(2*np.pi*frequencies[:,None]*t[None:,])
    Xall = np.zeros((N, K, 2)) #(time, frequency, num sinusoidal components)
    
    for n in range(N):
        c1 = C[:,n]
        s1 = S[:,n]
    
        Xn = np.column_stack([c1, s1])
        Xall[n,:,:] = Xn
    
    tss = np.sum((y - np.mean(y))**2)
    powers = []
    
    Xbat = Xall.transpose(1, 0, 2)
    XtX = np.einsum('kni,knj->kij', Xbat, Xbat)
    Xty = np.einsum('kni,n->ki', Xbat, y)
    
    beta = np.einsum('kij,kj->ki', np.linalg.pinv(XtX), Xty)
    yfit = np.einsum('kni,ki->kn', Xbat, beta)
    rss = np.sum((yfit - y[None,:])**2, axis=1)
    powers = 1 - rss / tss
    
    return frequencies, powers

def refine_single_frequency(t, y, w0, window=None):
    '''
    Locally refine a single-sinusoid frequency estimate w0 (e.g. a
    Lomb-Scargle periodogram grid peak) via least-squares optimization,
    within a narrow window around w0 so this polishes the given peak
    rather than re-searching (and possibly jumping to a different peak).
    '''
    t = np.asarray(t)
    y = np.asarray(y)
    w0 = float(np.atleast_1d(w0).ravel()[0])
    if window is None:
        # A few periodogram grid steps' worth, i.e. a few multiples of the
        # baseline-limited frequency resolution -- enough to polish the
        # peak, not enough to wander into a different one.
        window = 8*2*np.pi/(t.max() - t.min())

    def rss(w):
        w = w[0]
        X = np.column_stack([np.cos(w*t), np.sin(w*t)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.sum((y - X @ beta)**2)

    lo, hi = max(0.0, w0 - window), w0 + window
    res = minimize(rss, x0=[w0], method='Nelder-Mead', bounds=[(lo, hi)],
                    options={'xatol': 1e-10, 'fatol': 1e-12})
    return res.x[0]


def find_double_peak(t, y, refine=True):
    '''
    Find the two dominant frequencies via sequential prewhitening:
      1. Find the strongest periodicity with a standard Lomb-Scargle
         periodogram (astropy's autopower, which scales its frequency
         resolution to the data's time baseline automatically).
      2. Subtract that best-fit sinusoid from the data.
      3. Run a second periodogram on the residual to find the second
         frequency, at the same baseline-appropriate resolution.
      4. (Optional) a short local joint refinement of (w1, w2) to squeeze
         out the best joint least-squares fit near this starting point.
    '''
    t = np.asarray(t)
    y = np.asarray(y)

    freq1, power1 = LombScargle(t, y).autopower()
    w1_hat = freq1[np.argmax(power1)]*2*np.pi

    X1 = np.column_stack([np.cos(w1_hat*t), np.sin(w1_hat*t)])
    beta1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta1

    freq2, power2 = LombScargle(t, resid).autopower()
    w2_hat = freq2[np.argmax(power2)]*2*np.pi

    if refine:
        def rss(params):
            w1, w2 = params
            X = np.column_stack([np.cos(w1*t), np.sin(w1*t), np.cos(w2*t), np.sin(w2*t)])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            return np.sum((y - X @ beta)**2)

        res = minimize(rss, x0=[w1_hat, w2_hat], method='Nelder-Mead',
                        options={'xatol': 1e-8, 'fatol': 1e-10})
        if res.success:
            w1_hat, w2_hat = res.x

    return w1_hat, w2_hat


def double_Lomb_Scargle(t, y, K=12500):
    N = len(t)
    frequencies = np.linspace(0, 1/np.min(t), K)
    frequencies = np.linspace(0, 1.5, K)
    C = np.cos(2*np.pi*frequencies[:,None]*t[None:,])
    S = np.sin(2*np.pi*frequencies[:,None]*t[None:,])
    Xall = np.zeros((N, K**2, 4)) #(time, frequency, num sinusoidal components)
    
    for n in range(N):
        c1 = np.tile(C[:,n], K)
        s1 = np.tile(S[:,n], K)
        c2 = np.repeat(C[:,n], K)
        s2 = np.repeat(S[:,n], K)
        
        diag_mask = np.repeat(np.arange(K), K) == np.tile(np.arange(K), K)
        c2[diag_mask] = 0
        s2[diag_mask] = 0
    
        Xn = np.column_stack([c1, s1, c2, s2])
        Xall[n,:,:] = Xn
    
    tss = np.sum((y - np.mean(y))**2)
    powers = []
    
    Xbat = Xall.transpose(1, 0, 2)
    XtX = np.einsum('kni,knj->kij', Xbat, Xbat)
    Xty = np.einsum('kni,n->ki', Xbat, y)
    
    beta = np.einsum('kij,kj->ki', np.linalg.pinv(XtX), Xty)
    yfit = np.einsum('kni,ki->kn', Xbat, beta)
    rss = np.sum((yfit - y[None,:])**2, axis=1)
    powers = 1 - rss / tss
    
    powers = powers.reshape(K, K)
    return frequencies, powers