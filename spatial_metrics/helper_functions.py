import numpy as np
from scipy.io import loadmat
import pandas as pd
import os
from scipy import stats as stats


# implementing eegfilt
def eegfilt(LFP, fs, lowcut, highcut,order=3):
    from scipy import signal
    import numpy as np

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    if low == 0:
        b, a = signal.butter(order, high, btype='low')
        filtered = signal.filtfilt(b, a, LFP)
    elif high == 0:
        b, a = signal.butter(order, low, btype='high')
        filtered = signal.filtfilt(b, a, LFP) 
    else:
        b, a = signal.butter(order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, LFP)
        
        
    return filtered


def smooth(x,window_len=11,window='hanning'):
    
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[int(window_len/2-1):-int(window_len/2)]




def gaussian_smooth_2d(input_matrix,s_in):
    import numpy as np
    from scipy import signal as sig

    gaussian2d = gaussian2d_kernel(s_in)
    
    
    smoothed_matrix = sig.convolve2d(input_matrix,gaussian2d,mode='same')
    
    
    return smoothed_matrix

def gaussian2d_kernel(s):
    import numpy as np
    x_vec = np.arange(-100,101,1)
    y_vec = np.arange(-100,101,1)
#     s = 2
    gaussian_kernel = np.zeros([y_vec.shape[0],x_vec.shape[0]])
    x_count = 0
    for xx in x_vec:
        y_count = 0
        for yy in y_vec:
            gaussian_kernel[y_count,x_count] = np.exp(-((xx**2 + yy**2)/(2*(s**2))))

            y_count += 1
        x_count += 1

    return gaussian_kernel
