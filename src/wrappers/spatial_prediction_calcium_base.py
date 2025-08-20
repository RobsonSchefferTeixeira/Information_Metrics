import numpy as np
import os
import warnings

from src.utils import helper_functions as helper
from src.utils import surrogate_functions as surrogate
from src.utils import information_base as info
from src.utils import smoothing_functions as smooth

from src.utils.validators import ParameterValidator,DataValidator
import src.utils.bootstrapped_estimation as be
import src.utils.decoding_model_helper_functions as decoder_helper
import src.utils.decoders as decoder

from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed


class SpatialPrediction:
    def __init__(self,**kwargs):
           
        kwargs.setdefault('num_of_folds', 10)
        kwargs.setdefault('classifier_parameters',{'gaussian_nb': {'priors': 'uniform'}})

        kwargs.setdefault('signal_type',None)
        kwargs.setdefault('animal_id', None)
        kwargs.setdefault('day', None)
        kwargs.setdefault('neuron', None)
        kwargs.setdefault('trial', None)
        kwargs.setdefault('dataset', None)

        kwargs.setdefault('saving_path', os.getcwd())
        kwargs.setdefault('saving', False)
        kwargs.setdefault('saving_string', 'SpatialMetrics')
        kwargs.setdefault('overwrite', False)

        kwargs.setdefault('min_time_spent', 0.1)
        kwargs.setdefault('min_visits', 1)
        kwargs.setdefault('min_speed_threshold', 2.5)
        kwargs.setdefault('speed_smoothing_sigma', 1)
        kwargs.setdefault('x_bin_size', 1)
        kwargs.setdefault('y_bin_size', None)
        kwargs.setdefault('environment_edges', None)
        kwargs.setdefault('map_smoothing_sigma_x', 2)
        kwargs.setdefault('map_smoothing_sigma_y', 2)
        kwargs.setdefault('shift_time', 10)
        kwargs.setdefault('num_cores', 1)
        kwargs.setdefault('num_surrogates', 200)
        kwargs.setdefault('nbins_cal', 10)


        valid_kwargs = ['num_of_folds','classifier_parameters','signal_type','animal_id', 'day', 'neuron', 'dataset', 'trial',
                        'min_time_spent', 'min_visits', 'min_speed_threshold', 'speed_smoothing_sigma',
                        'x_bin_size', 'y_bin_size', 'shift_time', 'map_smoothing_sigma_x','map_smoothing_sigma_y','num_cores',
                        'num_surrogates', 'saving_path', 'saving', 'saving_string', 'environment_edges', 'nbins_cal','overwrite']
        

        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            setattr(self, k, v)

        ParameterValidator.validate_all(kwargs)

        self.__dict__['input_parameters'] = kwargs

    def main(self, signal_data):

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        filename = helper.filename_constructor(self.saving_string, self.animal_id, self.dataset, self.day, self.neuron, self.trial)
        full_path = f"{self.saving_path}/{filename}"
        # Check if the file exists and handle based on the overwrite flag
        if os.path.exists(full_path) and not self.overwrite:
            print(f"File already exists and overwrite is set to False: {full_path}")
            return
            
        if  DataValidator.is_empty_or_all_nan(signal_data.input_signal) or DataValidator.is_empty_or_all_nan(signal_data.x_coordinates):
            warnings.warn("Signal is constant, contains only NaN's or is empty", UserWarning)
            inputdict = np.nan
        
        elif np.allclose(signal_data.input_signal, signal_data.input_signal[0],equal_nan=True):
            warnings.warn("Signal is constant", UserWarning)
            inputdict = np.nan
       
        else:
            
            if signal_data.speed is None:
                signal_data.add_speed(self.speed_smoothing_sigma)

            x_grid, y_grid, x_center_bins, y_center_bins, x_center_bins_repeated, y_center_bins_repeated = helper.get_position_grid(
                signal_data.x_coordinates, signal_data.y_coordinates, self.x_bin_size, self.y_bin_size,
                environment_edges=signal_data.environment_edges)

            signal_data.add_position_binned(x_grid, y_grid)

            signal_data.add_visits(x_center_bins, y_center_bins)

            signal_data.add_position_time_spent()

            DataValidator.get_valid_timepoints(signal_data, self.min_speed_threshold, self.min_visits, self.min_time_spent)

            signal_data.add_peaks_detection()
            
            signal_data.add_binned_input_signal(self.nbins_cal)

            X = helper.ensure_2d_row(signal_data.input_signal.copy()).T
            
            y = signal_data.position_binned.copy()

            classifier_name, classifier_kwargs = next(iter(self.classifier_parameters.items()))

            dl = decoder.DecoderLearner(scale_data=True)

            y_pred = dl.run_classifier(X, y, kfolds = self.num_of_folds, decoder=classifier_name, classifier_params = classifier_kwargs)

            lookup = self.build_bin_label_to_coord_map(y, signal_data.bin_coordinates)

            coords_y_pred = np.array([lookup.get(b, np.nan) for b in y_pred])

            continuous_error, mean_error = self.get_continuous_error(coords_y_pred, signal_data.x_coordinates, signal_data.y_coordinates)

            spatial_error, spatial_error_smoothed = self.get_spatial_error(continuous_error, signal_data.x_coordinates, x_grid, self.map_smoothing_sigma_x,
                                signal_data.y_coordinates, y_grid, self.map_smoothing_sigma_y)
            
            results = self.parallelize_surrogate(X,y,self.num_of_folds,self.classifier_parameters,
                                                 signal_data.x_coordinates,signal_data.y_coordinates,
                                                 signal_data.bin_coordinates,
                                                 x_grid, y_grid,
                                                 signal_data.sampling_rate, self.shift_time, self.num_cores, self.num_surrogates)

            '''
            events_error = []
            for peaks in signal_data.peaks_idx:
                events_error.extend(continuous_error[peaks])
            events_error = np.array(events_error)
            mean_events_error = np.nanmean(events_error)
            '''
            
            events_error = []
            for peaks in signal_data.peaks_idx:
                events_error.append(continuous_error[peaks])
            mean_events_error = np.nanmean(np.concatenate(events_error))
            

            mean_error_shifted = []
            spatial_error_shifted = []
            mean_events_error_shifted = []
            for perm in range(self.num_surrogates):
                mean_error_shifted.append(results[perm][0])
                spatial_error_shifted.append(results[perm][1])
                continuous_error_shifted = results[perm][2]

                mean_events_error_shifted_aux = []
                for peaks in signal_data.peaks_idx:
                    mean_events_error_shifted_aux.append(continuous_error_shifted[peaks])
                mean_events_error_shifted.append(np.nanmean(np.concatenate(mean_events_error_shifted_aux)))

            mean_error_shifted = np.array(mean_error_shifted)
            spatial_error_shifted = np.array(spatial_error_shifted)
            mean_events_error_shifted = np.array(mean_events_error_shifted)

            mean_error_zscored, mean_error_centered = info.get_mutual_information_zscored(mean_error, mean_error_shifted)
            mean_error_statistic = be.calculate_p_value(mean_error, mean_error_shifted, alternative='less')
            mean_error_pvalue = mean_error_statistic.p_value

            mean_events_error_zscored, mean_events_error_centered = info.get_mutual_information_zscored(mean_events_error, mean_events_error_shifted)
            mean_events_error_statistic = be.calculate_p_value(mean_events_error, mean_events_error_shifted, alternative='less')
            mean_events_error_pvalue = mean_events_error_statistic.p_value


            inputdict = dict()
            
            inputdict['continuous_error'] = continuous_error
            inputdict['mean_error'] = mean_error
            inputdict['mean_error_zscored'] = mean_error_zscored
            inputdict['mean_error_centered'] = mean_error_centered
            inputdict['mean_error_shifted'] = mean_error_shifted     
            inputdict['mean_error_pvalue'] = mean_error_pvalue     
 
            inputdict['events_error'] = events_error

            inputdict['mean_events_error'] = mean_events_error
            inputdict['mean_events_error_zscored'] = mean_events_error_zscored     
            inputdict['mean_events_error_centered'] = mean_events_error_centered  
            inputdict['mean_events_error_shifted'] = mean_events_error_shifted    
            inputdict['mean_events_error_pvalue'] = mean_events_error_pvalue     

            inputdict['spatial_error'] = spatial_error
            inputdict['spatial_error_shifted'] = spatial_error_shifted

            inputdict['spatial_error_smoothed'] = spatial_error_smoothed
            inputdict['x_grid'] = x_grid
            inputdict['y_grid'] = y_grid
            inputdict['x_center_bins'] = x_center_bins
            inputdict['y_center_bins'] = y_center_bins

            inputdict['input_parameters'] = self.__dict__['input_parameters']

            filename = helper.filename_constructor(self.saving_string,self.animal_id,self.dataset,self.day,self.neuron,self.trial)

            if self.saving == True:
                helper.caller_saving(inputdict,filename,self.saving_path, self.overwrite)
            else:
                print('File not saved!')

        return inputdict
        
    

    def parallelize_surrogate(self, X, y, kfolds, classifier_parameters, x_coordinates, y_coordinates, bin_coordinates, x_grid, y_grid,
                            sampling_rate, shift_time, num_cores, num_surrogates):
        

        with tqdm_joblib(tqdm(desc="Processing Surrogates", total=num_surrogates)) as progress_bar:
            results = Parallel(n_jobs=num_cores)(
                delayed(self.run_classifier_surrogate)
                (
                    X, y, kfolds, classifier_parameters, x_coordinates, y_coordinates, bin_coordinates, x_grid, y_grid, sampling_rate, shift_time
                )
                for _ in range(num_surrogates)
            )
        return results


    def run_classifier_surrogate(self, X, y, kfolds, classifier_parameters, x_coordinates, y_coordinates, bin_coordinates, x_grid, y_grid, sampling_rate, shift_time):
        
        classifier_name, classifier_kwargs = next(iter(classifier_parameters.items()))

        dl = decoder.DecoderLearner(scale_data=True)

        X_shifted = surrogate.get_signal_surrogate(X, sampling_rate, shift_time, axis=0)

        y_pred_shifted = dl.run_classifier(X_shifted, y, kfolds, decoder=classifier_name, classifier_params = classifier_kwargs)

        lookup = self.build_bin_label_to_coord_map(y, bin_coordinates)
        
        coords_y_pred_shifted = np.array([lookup.get(b, np.nan) for b in y_pred_shifted])

        continuous_error_shifted,mean_error_shifted = self.get_continuous_error(coords_y_pred_shifted, x_coordinates, y_coordinates)

        spatial_error_shifted, spatial_error_smoothed_shifted = self.get_spatial_error(continuous_error_shifted, x_coordinates, x_grid, self.map_smoothing_sigma_x,
                                y_coordinates, y_grid, self.map_smoothing_sigma_y)


        return mean_error_shifted, spatial_error_shifted, continuous_error_shifted




    def build_bin_label_to_coord_map(self, position_binned, bin_coordinates):
        """
        Build a dictionary mapping bin indices to center coordinates.

        Parameters
        ----------
        position_binned : np.ndarray
            Array of integers representing bin labels per sample.
        bin_coordinates : np.ndarray
            Array of bin center coordinates.
            Shape can be (n_samples,) for 1D or (n_samples, 2) for 2D.

        Returns
        -------
        dict
            Dictionary mapping each bin label to its center coordinate(s).
        """

        unique_bins = np.unique(position_binned[~np.isnan(position_binned)]).astype(int)
        lookup = {}
        is_2d = bin_coordinates.ndim == 2
        for b in unique_bins:
            mask = position_binned == b
            coord = bin_coordinates[mask,:][0,:].copy() if is_2d else bin_coordinates[mask][0].copy()
            lookup[b] = coord
                            
        return lookup


    
    def get_continuous_error(self, coords_y_pred, x_coordinates, y_coordinates):
        """
        Compute Euclidean distance errors between predicted and actual coordinates.

        Parameters
        ----------
        coords_y_pred : np.ndarray
            Predicted coordinates per sample. Shape (n_samples,) for 1D or (n_samples, 2) for 2D.
        x_coordinates : np.ndarray
            Actual x-coordinates.
        y_coordinates : np.ndarray or None
            Actual y-coordinates. If None, computes 1D error.

        Returns
        -------
        continuous_error : np.ndarray
            Euclidean distance errors for each prediction.
        mean_error : float
            Mean of the errors, ignoring NaNs.
        """
        if coords_y_pred.ndim > 1:
            diffx = (coords_y_pred[:, 0] - x_coordinates) ** 2
            diffy = (coords_y_pred[:, 1] - y_coordinates) ** 2
        else:
            diffx = (coords_y_pred - x_coordinates) ** 2
            diffy = 0

        continuous_error = np.sqrt(diffx + diffy)
        mean_error = np.nanmean(continuous_error)
        return continuous_error, mean_error




    def get_spatial_error(self, continuous_error, x_coordinates, x_grid, sigma_x=1.0,
                    y_coordinates=None, y_grid=None, sigma_y=None):
        """
        Computes a spatial error map from decoding errors and applies Gaussian smoothing.

        Parameters
        ----------
        continuous_error : np.ndarray
            Decoding errors for each sample.
        x_coordinates : np.ndarray
            X-coordinates of the animal's position.
        x_grid : np.ndarray
            Bin edges for the x-dimension.
        sigma_x : float
            Smoothing standard deviation in spatial units (x).
        y_coordinates : np.ndarray, optional
            Y-coordinates of the animal's position.
        y_grid : np.ndarray, optional
            Bin edges for the y-dimension.
        sigma_y : float, optional
            Smoothing standard deviation in spatial units (y). Defaults to sigma_x.

        Returns
        -------
        spatial_error_map : np.ndarray
            Raw average error map.
        spatial_error_map_smoothed : np.ndarray
            Smoothed error map using Gaussian kernel.
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        if y_coordinates is None:
            # 1D mode
            spatial_error_map = np.full(len(x_grid) - 1, np.nan)
            for xx in range(len(x_grid) - 1):
                in_x_bin = (x_coordinates >= x_grid[xx]) & (x_coordinates < x_grid[xx + 1])
                spatial_error_map[xx] = np.nanmean(continuous_error[in_x_bin])

            sigma_x_points = smooth.get_sigma_points(sigma_x, x_grid)
            kernel, _ = smooth.generate_1d_gaussian_kernel(sigma_x_points, truncate=4.0)
            spatial_error_map_smoothed = smooth.gaussian_smooth_1d(spatial_error_map, kernel, handle_nans=False)

        else:
            # 2D mode
            if sigma_y is None:
                sigma_y = sigma_x

            spatial_error_map = np.full((len(y_grid) - 1, len(x_grid) - 1), np.nan)
            for xx in range(len(x_grid) - 1):
                for yy in range(len(y_grid) - 1):
                    in_x_bin = (x_coordinates >= x_grid[xx]) & (x_coordinates < x_grid[xx + 1])
                    in_y_bin = (y_coordinates >= y_grid[yy]) & (y_coordinates < y_grid[yy + 1])
                    in_bin = in_x_bin & in_y_bin
                    spatial_error_map[yy, xx] = np.nanmean(continuous_error[in_bin])

            sigma_x_points = smooth.get_sigma_points(sigma_x, x_grid)
            sigma_y_points = smooth.get_sigma_points(sigma_y, y_grid)

            kernel, _ = smooth.generate_2d_gaussian_kernel(
                sigma_x_points, sigma_y_points, radius_x=None, radius_y=None, truncate=4.0
            )
            spatial_error_map_smoothed = smooth.gaussian_smooth_2d(spatial_error_map, kernel, handle_nans=False)

        return spatial_error_map, spatial_error_map_smoothed

