import numpy as np
import numpy.random as random

def generate_time(period_days, window_jitter_hours=1.0, jitter_window=True):
    '''
    Generates irregularly spaced time stamps.
    '''
    #Get base observations
    t_max = 5*period_days
    
    if period_days < 0.5:
        cadence = 10/1440  #Every 10 minutes
    elif period_days < 5.0:
        cadence = 30/1440  #Every 30 minutes
    elif period_days < 100.0:
        cadence = 2/24     #Every 2 hours
    else:
        cadence = 1.0        #Every 1 day

    t_uniform = np.arange(cadence,t_max,cadence)

    #Mask daylight hours
    time = t_uniform % 1
    if jitter_window:
        # Independent random offset to the morning/evening cutoff for each calendar day, so the window function is not exactly periodic.
        day_index = np.floor(t_uniform).astype(int)
        n_days = day_index.max() + 1
        jitter = window_jitter_hours/24
        morning_offset = np.random.uniform(-jitter, jitter, size=n_days)[day_index]
        evening_offset = np.random.uniform(-jitter, jitter, size=n_days)[day_index]
        night = (time >= 0.75 + evening_offset) | (time <= 0.25 + morning_offset)
    else:
        night = (time >= 0.75) | (time <= 0.25) #Mask points from 6am-6pm
    t_night = t_uniform[night]

    #Randomly mask certain observations
    mask = mask = np.random.choice(range(0, len(t_night)), size=round(len(t_night)/4), replace=False) #25% based on Mauna Kea cloud cover (See Steinbring et. al., 2009)

    t = np.delete(t_night, mask)
    

    return t

#References
#Steinbring, E., Cuillandre, J.-C., and Magnier, E. (2009). Mauna Kea Sky Transparency from CFHT SkyProbe Data. Publications of the Astronomical Society of the Pacific, 121(877):295.


