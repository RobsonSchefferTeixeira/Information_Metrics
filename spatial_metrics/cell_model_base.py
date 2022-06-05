import numpy as np
import os
import numpy as np
import math
import random
from scipy import interpolate

def generate_random_walk_old(input_srate = 100.,input_total_Time = 500,rho1  = 1.,sigma = 0.02,mu_e  = 0.,smooth_coeff = 0.5):

    # global srate
    srate = float(np.copy(input_srate))
    
    # global total_Time
    total_Time = float(np.copy(input_total_Time))
    
    # global total_points
    total_points = int(total_Time*srate)
    
    y_coordinates    = np.zeros(total_points)
    x_coordinates    = np.zeros(total_points)

    epsy   = np.random.normal(mu_e,sigma,total_points) 
    epsx   = np.random.normal(mu_e,sigma,total_points) 


    for t in range(2,total_points):
        aux = rho1*y_coordinates[t-1] + epsy[t]
        if aux <= 1 and aux >= 0:
            y_coordinates[t] = aux
        else:
            y_coordinates[t] = y_coordinates[t-1]    

    for t in range(2,total_points):

        aux = rho1*x_coordinates[t-1] + epsx[t]
        if aux <= 1 and aux >= 0:
            x_coordinates[t] = aux
        else:
            x_coordinates[t] = x_coordinates[t-1]    

    x_coordinates = smooth(np.squeeze(x_coordinates),round_up_to_even(int(smooth_coeff*srate)))
    y_coordinates = smooth(np.squeeze(y_coordinates),round_up_to_even(int(smooth_coeff*srate)))

    timevector = np.linspace(0,total_points/srate,total_points)
    dt = 1/srate
    speed = np.sqrt(np.diff(x_coordinates)**2 + np.diff(y_coordinates)**2) / dt
    speed = np.hstack([speed,0])
    


    return x_coordinates,y_coordinates,speed,timevector

def round_up_to_even(f):
    return math.ceil(f / 2.) * 2

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


def generate_random_walk(input_srate = 100.,input_total_Time = 500,head_direction_srate = 10., speed_srate = 5., rho1  = 1.,sigma = 0.02,mu_e  = 0.,smooth_coeff = 0.5, **kwargs):

    # x_bin_size and y_bin_size in cm
    # environment_edges = [[x1, x2], [y1, y2]]
    environment_edges = kwargs.get('environment_edges')

    # global srate
    srate = float(np.copy(input_srate))
    
    # global total_Time
    total_Time = float(np.copy(input_total_Time))
    
    # global total_points
    total_points = int(total_Time*srate)
    

    total_points_head = int(total_Time*head_direction_srate)
    head_direction = np.zeros(total_points_head)

    head_direction_sigma = math.pi/4
    head_direction_mu = 0
    randomphases = np.random.normal(head_direction_mu,head_direction_sigma,total_points_head)

    for t in range(1,total_points_head):
        head_direction[t] = np.angle(np.exp(1j*(head_direction[t-1] + randomphases[t-1])))


    y_original = head_direction.copy()
    x_original = np.linspace(0,total_Time,head_direction.shape[0])
    interpol_func = interpolate.interp1d(x_original,y_original,kind = 'cubic')
    x_new = np.linspace(0,total_Time,total_points)
    head_direction_new = interpol_func(x_new)


    total_points_spd = int(total_Time*speed_srate)
    speeds = np.zeros(total_points_spd)
    randomspeeds = np.random.exponential(100./srate,total_points_spd)
    for t in range(1,total_points_spd):
        speeds[t] = randomspeeds[t-1]

    y_original = speeds
    x_original = np.linspace(0,total_Time,speeds.shape[0])
    interpol_func = interpolate.interp1d(x_original,y_original,kind = 'cubic')
    x_new = np.linspace(0,total_Time,total_points)
    speeds_new = interpol_func(x_new)


    y_coordinates    = np.zeros(total_points)
    x_coordinates    = np.zeros(total_points)

    epsy   = np.random.normal(mu_e,sigma,total_points) 
    epsx   = np.random.normal(mu_e,sigma,total_points) 

    for t in range(1,total_points):

        y_coordinates[t] = y_coordinates[t-1] + speeds_new[t]*np.sin(head_direction_new[t]) + rho1*epsy[t]
        x_coordinates[t] = x_coordinates[t-1] + speeds_new[t]*np.cos(head_direction_new[t]) + rho1*epsx[t]

        if environment_edges:
            if y_coordinates[t] > environment_edges[1][1] or y_coordinates[t] < environment_edges[1][0] or x_coordinates[t] > environment_edges[0][1] or x_coordinates[t] < environment_edges[0][0]:

                head_direction_new = head_direction_new+math.pi

                y_coordinates[t] = y_coordinates[t-1] + speeds_new[t]*np.sin(head_direction_new[t]) + rho1*epsy[t]
                x_coordinates[t] = x_coordinates[t-1] + speeds_new[t]*np.cos(head_direction_new[t]) + rho1*epsx[t]
        else:
            
            if y_coordinates[t] > 100 or y_coordinates[t] < 0 or x_coordinates[t] > 100 or x_coordinates[t] < 0:
                head_direction_new = head_direction_new+math.pi

                y_coordinates[t] = y_coordinates[t-1] + speeds_new[t]*np.sin(head_direction_new[t]) + rho1*epsy[t]
                x_coordinates[t] = x_coordinates[t-1] + speeds_new[t]*np.cos(head_direction_new[t]) + rho1*epsx[t]

    x_coordinates[x_coordinates < environment_edges[0][0]] = environment_edges[0][0]
    x_coordinates[x_coordinates > environment_edges[0][1]] = environment_edges[0][1]

    y_coordinates[y_coordinates < environment_edges[1][0]] = environment_edges[1][0]
    y_coordinates[y_coordinates > environment_edges[1][1]] = environment_edges[1][1]

    x_coordinates = smooth(np.squeeze(x_coordinates),round_up_to_even(int(smooth_coeff*srate)))
    y_coordinates = smooth(np.squeeze(y_coordinates),round_up_to_even(int(smooth_coeff*srate)))
    

    timevector = np.linspace(0,total_Time,total_points)
    dt = 1/srate
    speed = np.sqrt(np.diff(x_coordinates)**2 + np.diff(y_coordinates)**2) / dt
    speed = np.hstack([speed,0])
    
    return x_coordinates,y_coordinates,speed,timevector



def generate_arrivals(_lambda = 0.5,total_Time=100):

    _num_arrivals = int(_lambda*total_Time)
    _arrival_time = 0
    All_arrival_time = []

    for i in range(_num_arrivals):
        #Get the next probability value from Uniform(0,1)
        p = random.random()

        #Plug it into the inverse of the CDF of Exponential(_lambda)
        _inter_arrival_time = -math.log(1.0 - p)/_lambda

        #Add the inter-arrival time to the running sum
        _arrival_time = _arrival_time + _inter_arrival_time
        All_arrival_time.append(_arrival_time)
    All_arrival_time = np.array(All_arrival_time)
    All_arrival_time = All_arrival_time[All_arrival_time < total_Time]

    return All_arrival_time

def get_bins_edges(x_coordinates,y_coordinates,x_nbins,y_nbins):
    
    x_bins = np.linspace(np.nanmin(x_coordinates),np.nanmax(x_coordinates),x_nbins)
    y_bins = np.linspace(np.nanmin(y_coordinates),np.nanmax(y_coordinates),y_nbins)
    
    return x_bins,y_bins

def gaussian_kernel_2d(x_coordinates,y_coordinates,x_nbins=100,y_nbins=100,x_center = 0.5,y_center = 0.5, s = 0.1):
    x_bins,y_bins = get_bins_edges(x_coordinates,y_coordinates,x_nbins,y_nbins)
    
    gaussian_kernel = np.zeros([y_bins.shape[0],x_bins.shape[0]])
    x_count = 0
    for xx in x_bins:
        y_count = 0
        for yy in y_bins:
            gaussian_kernel[y_count,x_count] = np.exp(-(((xx - x_center)**2 + (yy-y_center)**2)/(2*(s**2))))
            y_count += 1
        x_count += 1
    
    return gaussian_kernel


def digitize_spiketimes(x_coordinates,y_coordinates,I_timestamps,x_nbins=100,y_nbins=100,x_center = 0.5,y_center = 0.5, s = 0.1):
    
    x_bins,y_bins = get_bins_edges(x_coordinates,y_coordinates,x_nbins,y_nbins)

    x_digitized = np.digitize(x_coordinates[I_timestamps],x_bins)-1
    y_digitized = np.digitize(y_coordinates[I_timestamps],y_bins)-1

    gaussian_kernel = gaussian_kernel_2d(x_coordinates,y_coordinates,x_nbins,y_nbins,x_center,y_center, s)
    
    modulated_timestamps = []
    for spk in range(0,x_digitized.shape[0]):
        random_number = random.choices([0,1], [1-gaussian_kernel[y_digitized[spk],x_digitized[spk]],gaussian_kernel[y_digitized[spk],x_digitized[spk]]])[0]
        if random_number == 1:
            modulated_timestamps.append(I_timestamps[spk])
    modulated_timestamps = np.array(modulated_timestamps)
    return modulated_timestamps


def generate_calcium_signal(modulated_timestamps,total_points,srate,noise_level = 0.01, b = 5.):

    dt = 1/srate
    timevector = np.linspace(0,total_points/srate,total_points)

    I_pf_timestamps = (modulated_timestamps).astype(int)

    All_arrival_continuous = np.zeros(timevector.shape[0])
    All_arrival_continuous[I_pf_timestamps] = 1
    a = 1.
    
    x = np.arange(0,5,dt)
    kernel = a*np.exp(-b*x)

    calcium_imag = np.convolve(kernel,All_arrival_continuous,mode='full')
    calcium_imag = np.copy(calcium_imag[0:total_points])
    calcium_imag = calcium_imag + noise_level*np.random.normal(0,1,calcium_imag.shape)
    
    return calcium_imag,timevector

