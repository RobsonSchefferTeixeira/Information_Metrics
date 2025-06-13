
import numpy as np
from src.utils import helper_functions as hf
from src.utils.validators import DataValidator
from src.utils import information_base as info

import warnings

class ProcessData:

    def __init__(self, input_signal, x_coordinates, y_coordinates, time_vector, sampling_rate = None, environment_edges = None, speed = None, coordinates_interpolation = False):
        
        self.input_signal = input_signal
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.time_vector = time_vector
        self.speed = speed
        self.sampling_rate = sampling_rate
        self.environment_edges = environment_edges

        if self.sampling_rate is None:
            self.sampling_rate = 1 / np.nanmean(np.diff(self.time_vector))

        
        DataValidator.validate_environment_edges(self)
        DataValidator.initial_setup(self) # Initial Setup and Conversion
        DataValidator.validate_and_correct_shape(self) # Shape Validation and Correction
        self.x_coordinates, self.y_coordinates = hf.correct_coordinates(self.x_coordinates,self.y_coordinates,environment_edges=self.environment_edges)

        if coordinates_interpolation:
            self.x_coordinates, self.y_coordinates = hf.correct_lost_tracking(self.x_coordinates, self.y_coordinates, self.time_vector, self.sampling_rate, min_epoch_length=0.5)
        
        DataValidator.filter_invalid_values(self) # NaN/Infinite Value Filtering



    def add_speed(self,speed_smoothing_sigma):

        if self.speed is None:
            self.speed,self.speed_smoothed = hf.get_speed(self.x_coordinates, self.y_coordinates, self.time_vector, speed_smoothing_sigma)
        else:
            _,self.speed_smoothed = hf.get_speed(self.x_coordinates, self.y_coordinates, self.time_vector, speed_smoothing_sigma)


    def add_position_binned(self,x_grid, y_grid):
        self.position_binned, self.bin_coordinates = hf.get_binned_position(self.x_coordinates, x_grid, self.y_coordinates, y_grid)

    def add_visits(self, x_center_bins, y_center_bins):
        self.visits_bins, self.new_visits_times = hf.get_visits(self.x_coordinates, self.position_binned, x_center_bins, self.y_coordinates, y_center_bins)

    def add_position_time_spent(self):
        self.time_spent_inside_bins = hf.get_position_time_spent(self.position_binned, self.sampling_rate)

    def add_binned_input_signal(self,nbins_cal, signal_type='raw'):
        if signal_type == 'binary':
            self.input_signal_binned = self.input_signal.copy()
        else:
            self.input_signal_binned = info.get_binned_signal(self.input_signal, nbins_cal)


    def add_peaks_detection(self, signal_type):
        signal = self.input_signal.copy()
        signal = np.atleast_2d(signal)  # Ensure 2D shape for uniform handling


        self.peaks_idx = []
        self.peaks_amplitude = []
        self.peaks_x_location = []
        self.peaks_y_location = []

        for row in signal:
            if signal_type == 'binary':
                peaks = row == 1
            else:
                peaks = hf.detect_peaks(
                    row,
                    mpd=0.5 * self.sampling_rate,
                    mph=1. * np.nanstd(row)
                )

            if peaks.size > 0:
                self.peaks_idx.append(peaks)
                self.peaks_amplitude.append(row[peaks])
                self.peaks_x_location.append(self.x_coordinates[peaks])
                if self.y_coordinates is None:
                    self.peaks_y_location.append(np.zeros(peaks.shape[0])+np.nan)
                else:
                    self.peaks_y_location.append(self.y_coordinates[peaks])
            else:
                self.peaks_idx.append([])
                self.peaks_amplitude.append([])
                self.peaks_x_location.append([])
                self.peaks_y_location.append([])



    @staticmethod
    def sync_imaging_to_video(input_signal, time_vector, track_time_vector):
        """
        Interpolates input_signal from time_vector to track_time_vector.
        Warns if track_time_vector is outside the time range of time_vector.

        Parameters:
        - input_signal (np.ndarray): Original signal values.
        - time_vector (np.ndarray): Time in seconds for each input_signal point.
        - track_time_vector (np.ndarray): New time vector to interpolate onto.

        Returns:
        - track_input_signal (np.ndarray): Interpolated signal at track_time_vector.
        """

        if np.isnan(input_signal).any():
            warnings.warn("input_signal contains NaN values. These may propagate or distort interpolation.")

        if np.isnan(time_vector).any():
            warnings.warn("time_vector contains NaN values. This may break interpolation.")

        if np.isnan(track_time_vector).any():
            warnings.warn("track_time_vector contains NaN values. Output will have NaNs at those positions.")

        if track_time_vector[0] < time_vector[0]:
            warnings.warn("track_time_vector starts before the first time point in time_vector. Values are extrapolated.")

        if track_time_vector[-1] > time_vector[-1]:
            warnings.warn("track_time_vector ends after the last time point in time_vector. Values are extrapolated.")

        input_signal = np.atleast_2d(input_signal)
        # Perform interpolation
        track_input_signal = np.vstack([
            np.interp(track_time_vector, time_vector, signal)
            for signal in input_signal
        ])

        return np.squeeze(track_input_signal)
