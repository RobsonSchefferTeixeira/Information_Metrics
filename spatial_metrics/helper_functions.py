import numpy as np
from scipy.io import loadmat
import pandas as pd
import os
from scipy import stats as stats
import sys
from scipy import interpolate

def correct_lost_tracking(x_coordinates,y_coordinates,track_timevector,mean_video_srate,min_epoch_length = 1):
    
    

    x_coordinates_interpolated = x_coordinates.copy()
    y_coordinates_interpolated = y_coordinates.copy()

    I_nan_events = np.where(np.isnan(x_coordinates*y_coordinates))[0]
    I_start = np.where(np.diff(I_nan_events) > 1)[0] + 1

    all_epoch_length = []
    for cc in range(I_start.shape[0]-1):
        beh_index = np.arange(I_start[cc],I_start[cc+1])

        epoch_length = (beh_index.shape[0]/mean_video_srate)
        all_epoch_length.append(epoch_length)

        if epoch_length <= min_epoch_length:

            events_around_nan = I_nan_events[beh_index]
            events_to_replace = np.arange(events_around_nan[0]-1,events_around_nan[-1]+2)

            window_beg = events_to_replace[0]
            window_end = events_to_replace[-1]

            x_original = track_timevector[[window_beg,window_end]]
            y_original = x_coordinates[[window_beg,window_end]]
            x_new = track_timevector[window_beg:window_end+1]
            interpol_func = interpolate.interp1d(x_original,y_original,kind='slinear')
            y_new = interpol_func(x_new)

            x_coordinates_interpolated[events_to_replace] = y_new


            x_original = track_timevector[[window_beg,window_end]]
            y_original = y_coordinates[[window_beg,window_end]]
            x_new = track_timevector[window_beg:window_end+1]
            interpol_func = interpolate.interp1d(x_original,y_original,kind='slinear')
            y_new = interpol_func(x_new)

            y_coordinates_interpolated[events_to_replace] = y_new

    all_epoch_length = np.array(all_epoch_length)

    return x_coordinates_interpolated,y_coordinates_interpolated


def filename_constructor(saving_string,animal_id,dataset,day,neuron,trial):

    first_string =  saving_string
    animal_id_string = '.' + animal_id
    dataset_string = '.Dataset.' + dataset
    day_string = '.Day.' + str(day)
    neuron_string = '.Neuron.' + str(neuron)
    trial_string = '.Trial.' + str(trial)

    filename_checklist = np.array([first_string,animal_id, dataset, day, neuron, trial])
    inlcude_this = np.where(filename_checklist != None)[0]

    filename_backbone = [first_string, animal_id_string,dataset_string, day_string, neuron_string, trial_string]

    filename = ''.join([filename_backbone[i] for i in inlcude_this])

    return filename

def caller_saving(inputdict,filename,saving_path):
    os.chdir(saving_path)
    output = open(filename, 'wb') 
    np.save(output,inputdict)
    output.close()
    print('File saved.')

def identify_islands(input_array):

    row = input_array.shape[0]
    col = input_array.shape[1]
    count = 0
    input_array2 = np.copy(input_array)
    
    for i in range(row):
        for j in range(col):
            if input_array[i,j] == 1:
                count+=1
                dfs(input_array,input_array2,count,row,col,i,j)
                
    return input_array2

def dfs(input_array,input_array2,count,row,col,i,j):

    if input_array[i,j] == 0:
        return
    input_array[i,j] = 0
    input_array2[i,j] = count

    if i != 0:
        dfs(input_array,input_array2,count,row,col,i-1,j)

    if i != row-1:
        dfs(input_array,input_array2,count,row,col,i+1,j)

    if j != 0:
        dfs(input_array,input_array2,count,row,col,i,j-1)

    if j != col - 1:
        dfs(input_array,input_array2,count,row,col,i,j+1)

def field_coordinates(place_field,smoothing_size=1,field_threshold = 2):
    
    place_field_to_smooth = np.copy(place_field)
    place_field_to_smooth[np.isnan(place_field_to_smooth)] = 0 
    place_field_smoothed = gaussian_smooth_2d(place_field_to_smooth,smoothing_size)

    I_threshold = field_threshold*np.nanstd(place_field_smoothed)

    field_above_threshold = np.copy(place_field_smoothed)
    field_above_threshold[field_above_threshold < I_threshold] = 0

    field_above_threshold_binary = np.copy(place_field_smoothed)
    field_above_threshold_binary[field_above_threshold_binary < I_threshold] = 0
    field_above_threshold_binary[field_above_threshold_binary >= I_threshold] = 1


    if np.any(field_above_threshold_binary>0):
        sys.setrecursionlimit(10000)
        place_field_identity = identify_islands(np.copy(field_above_threshold_binary))
        islands_id = np.unique(place_field_identity)[1:]
        num_of_islands = islands_id.shape[0]

        islands_y_max = []
        islands_x_max = []

        for ii in islands_id:
            max_val = np.max(place_field_smoothed[(place_field_identity == ii)])
            I_y_max,I_x_max = np.where(place_field_smoothed == max_val)
            islands_y_max.append(I_y_max[0])
            islands_x_max.append(I_x_max[0])

        islands_x_max = np.array(islands_x_max)
        islands_y_max = np.array(islands_y_max)


    else:
        num_of_islands = 0
        islands_y_max = np.nan
        islands_x_max = np.nan

    return num_of_islands,islands_x_max,islands_y_max



def preprocess_signal(input_signal,mean_video_srate,signal_type,z_threshold = 2):
    
    filtered_signal = eegfilt(input_signal,mean_video_srate,0,2,order=2)
    diff_signal = np.hstack([np.diff(filtered_signal),0])
    
    if signal_type == 'Raw':
        output_signal = input_signal
        
    elif signal_type == 'Filtered':
        output_signal = filtered_signal

    elif signal_type == 'Diff':
        diff_signal_truncated = np.copy(diff_signal)
        diff_signal_truncated[diff_signal<0] = 0
        output_signal = diff_signal_truncated


    elif signal_type == 'Binarized':
        
        norm_signal = filtered_signal/np.std(filtered_signal)
        binarized_signal = np.zeros(diff_signal.shape[0])
        binarized_signal[(norm_signal>z_threshold) & (diff_signal>0)] = 1
        output_signal = binarized_signal

    else:
        raise NotImplementedError('Signal type unknown')
    
    return output_signal

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
