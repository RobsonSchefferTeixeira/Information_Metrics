
import numpy as np
from src.utils import helper_functions as hf
from src.utils.validators import DataValidator
from src.utils import information_base as info

import warnings

class ProcessData:

    def __init__(self, input_signal, x_coordinates, y_coordinates, time_vector, sampling_rate = None, environment_edges = None, speed = None):
        
        self.input_signal = input_signal
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.time_vector = time_vector
        self.speed = speed
        self.sampling_rate = sampling_rate

        if self.sampling_rate is None:
            self.sampling_rate = 1 / np.nanmean(np.diff(self.time_vector))

        
        DataValidator.validate_input_data(self)
        DataValidator.validate_environment_edges(self,environment_edges)


    def add_speed(self,speed_smoothing_sigma):

        if self.speed is None:
            self.speed,self.speed_smoothed = hf.get_speed(self.x_coordinates, self.y_coordinates, self.time_vector, speed_smoothing_sigma)
        else:
            _,self.speed_smoothed = hf.get_speed(self.x_coordinates, self.y_coordinates, self.time_vector, speed_smoothing_sigma)


    def add_position_binned(self,x_grid, y_grid):
        self.position_binned = hf.get_binned_position(self.x_coordinates, self.y_coordinates, x_grid, y_grid)

    def add_visits(self, x_center_bins, y_center_bins):
        self.visits_bins, self.new_visits_times = hf.get_visits(self.x_coordinates, self.y_coordinates, self.position_binned, x_center_bins, y_center_bins)

    def add_position_time_spent(self):
        self.time_spent_inside_bins = hf.get_position_time_spent(self.position_binned, self.sampling_rate)

    def add_binned_input_signal(self,nbins_cal):
        self.input_signal_binned = info.get_binned_signal(self.input_signal, nbins_cal)


    def add_peaks_detection(self,signal_type):
            if signal_type == 'binary':

                self.peaks_idx = self.input_signal == 1
                self.peaks_amplitude = self.input_signal[self.peaks_idx]
                self.peaks_x_location = self.x_coordinates[self.peaks_idx]
                self.peaks_y_location = self.y_coordinates[self.peaks_idx]

            else:

                self.peaks_idx = hf.detect_peaks(self.input_signal,mpd=0.5 * self.sampling_rate, mph=1. * np.nanstd(self.input_signal))
                self.peaks_amplitude = self.input_signal[self.peaks_idx]
                self.peaks_x_location = self.x_coordinates[self.peaks_idx]
                self.peaks_y_location = self.y_coordinates[self.peaks_idx]

            # else:
            #    warnings.warn(f"Unrecognized signal_type '{signal_type}'. Expected 'binary' or 'continuous'.", UserWarning)

