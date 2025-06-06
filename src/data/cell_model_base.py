import numpy as np
import os
import numpy as np
import math
import random
from scipy import interpolate
from src.utils import helper_functions as hf
from src.utils import smoothing_functions as smooth


def generate_random_walk_old(input_srate = 100.,input_total_Time = 500,rho1  = 1.,sigma = 0.02,mu_e  = 0.,smooth_coeff = 0.5):

    # global srate
    sampling_rate = float(np.copy(input_srate))
    
    # global total_Time
    total_Time = float(np.copy(input_total_Time))
    
    # global total_points
    total_points = int(total_Time*sampling_rate)
    
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

    x_coordinates = smooth(np.squeeze(x_coordinates),round_up_to_even(int(smooth_coeff*sampling_rate)))
    y_coordinates = smooth(np.squeeze(y_coordinates),round_up_to_even(int(smooth_coeff*sampling_rate)))

    timevector = np.linspace(0,total_points/sampling_rate,total_points)
    dt = 1/sampling_rate
    speed = np.sqrt(np.diff(x_coordinates)**2 + np.diff(y_coordinates)**2) / dt
    speed = np.hstack([speed,0])
    
    return x_coordinates,y_coordinates,speed,timevector

def round_up_to_even(f):
    return math.ceil(f / 2.) * 2


def generate_random_walk(input_srate=100., input_total_Time=500, head_direction_srate=10.,
                         speed_srate=5., rho1=1., sigma=0.02, mu_e=0., sigma_x=0.1, sigma_y = 0.1, **kwargs):


    environment_edges = kwargs.get('environment_edges', [[0, 100], [0, 100]])

    sampling_rate = float(input_srate)
    total_Time = float(input_total_Time)
    total_points = int(total_Time * sampling_rate)

    total_points_head = int(total_Time * head_direction_srate)
    head_direction = np.zeros(total_points_head)
    
    head_direction_sigma = math.pi / 4
    head_direction_mu = 0
    random_phases = np.random.normal(head_direction_mu, head_direction_sigma, total_points_head)

    for t in range(1, total_points_head):
        head_direction[t] = np.angle(np.exp(1j * (head_direction[t - 1] + random_phases[t - 1])))

    x_original = np.linspace(0, total_Time, head_direction.shape[0])
    interpol_func = interpolate.interp1d(x_original, head_direction, kind='cubic')
    x_new = np.linspace(0, total_Time, total_points)
    head_direction_new = interpol_func(x_new)

    total_points_spd = int(total_Time * speed_srate)
    speeds = np.random.exponential(100. / sampling_rate, total_points_spd)
    speeds_new = np.interp(x_new, np.linspace(0, total_Time, speeds.shape[0]), speeds)

    y_coordinates = np.zeros(total_points)
    x_coordinates = np.zeros(total_points)

    x_coordinates[0] = random.uniform(*environment_edges[0])
    y_coordinates[0] = random.uniform(*environment_edges[1])

    epsy = np.random.normal(mu_e, sigma, total_points)
    epsx = np.random.normal(mu_e, sigma, total_points)

    for t in range(1, total_points):
        y_coordinates[t] = y_coordinates[t - 1] + speeds_new[t] * np.sin(head_direction_new[t]) + rho1 * epsy[t]
        x_coordinates[t] = x_coordinates[t - 1] + speeds_new[t] * np.cos(head_direction_new[t]) + rho1 * epsx[t]

        # Ensure the animal stays within the environment
        if x_coordinates[t] >= environment_edges[0][1] or x_coordinates[t] <= environment_edges[0][0] \
            or y_coordinates[t] >= environment_edges[1][1] or y_coordinates[t] <= environment_edges[1][0]:
            
            # head_direction_new[t:] += math.pi
            head_direction_new[t:] = np.angle(np.exp(1j*(head_direction_new[t:] + math.pi)))
            
            y_coordinates[t] = y_coordinates[t-1] + speeds_new[t] * np.sin(head_direction_new[t])
            x_coordinates[t] = x_coordinates[t-1] + speeds_new[t] * np.cos(head_direction_new[t])

    time_vector = np.linspace(0, total_Time, total_points)

    sigma_x_points = smooth.get_sigma_points(sigma_x, time_vector)
    kernel, x_mesh = smooth.generate_1d_gaussian_kernel(sigma_x_points, truncate=4.0)
    x_coordinates = smooth.gaussian_smooth_1d(x_coordinates.squeeze(), kernel, handle_nans=False)

    sigma_y_points = smooth.get_sigma_points(sigma_y, time_vector)
    kernel, y_mesh = smooth.generate_1d_gaussian_kernel(sigma_y_points, truncate=4.0)
    y_coordinates = smooth.gaussian_smooth_1d(y_coordinates.squeeze(), kernel, handle_nans=False)

    np.clip(x_coordinates, *environment_edges[0], out=x_coordinates)
    np.clip(y_coordinates, *environment_edges[1], out=y_coordinates)

    speed, smoothed_speed = hf.get_speed(x_coordinates, y_coordinates, time_vector, speed_smoothing_sigma = sigma_x_points)

    return x_coordinates, y_coordinates, speed, smoothed_speed, time_vector



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


def generate_poisson_spikes(rate, duration):
    """
    Generate spiking activity based on a Poisson distribution using the inverse of the exponential CDF.

    Parameters:
        rate (float): Average firing rate in spikes per second.
        duration (float): Duration of the spike train in seconds.

    Returns:
        spike_times (numpy.ndarray): Array of spike times in seconds.
    """
    # Calculate the expected number of spikes in the given duration.
    expected_spikes = rate * duration

    # Generate spike times using the inverse of the exponential CDF.
    spike_times = []
    time = 0
    while time < duration:
        rand_value = np.random.uniform()
        inter_spike_interval = -np.log(1 - rand_value) / rate
        time += inter_spike_interval
        spike_times.append(time)

    # Remove any spike times that exceed the duration.
    spike_times = np.array([t for t in spike_times if t <= duration])

    return spike_times



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
    modulated_timestamps = np.array(modulated_timestamps).astype(int)
    return modulated_timestamps


def generate_calcium_signal(modulated_timestamps,total_points,sampling_rate,noise_level = 0.01, b = 5.):

    dt = 1/sampling_rate
    timevector = np.linspace(0,total_points/sampling_rate,total_points)

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

