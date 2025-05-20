
import numpy as np
import os
import warnings

from src.utils import helper_functions as hf
from src.utils import surrogate_functions as surrogate
from src.utils import information_base as info
from src.utils import smoothing_functions as smooth

from src.utils.validators import ParameterValidator,DataValidator
import src.utils.bootstrapped_estimation as be
import src.utils.decoding_model_helper_functions as decoding


from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed


class SpatialPrediction:
    def __init__(self,**kwargs):
           
        kwargs.setdefault('num_of_folds', 10)
        kwargs.setdefault('classifier', 'GaussianBayes')

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


        valid_kwargs = ['num_of_folds','classifier','signal_type','animal_id', 'day', 'neuron', 'dataset', 'trial',
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
        
        filename = hf.filename_constructor(self.saving_string, self.animal_id, self.dataset, self.day, self.neuron, self.trial)
        full_path = f"{self.saving_path}/{filename}"
        # Check if the file exists and handle based on the overwrite flag
        if os.path.exists(full_path) and not self.overwrite:
            print(f"File already exists and overwrite is set to False: {full_path}")
            return
        
        if np.all(np.isnan(signal_data.input_signal)):
            warnings.warn("Signal contains only NaN's")
            inputdict = np.nan

        else:
            signal_data.x_coordinates, signal_data.y_coordinates = hf.correct_coordinates(signal_data.x_coordinates,signal_data.y_coordinates,environment_edges=signal_data.environment_edges)


            if signal_data.speed is None:
                signal_data.add_speed(self.speed_smoothing_sigma)

            x_grid, y_grid, x_center_bins, y_center_bins, x_center_bins_repeated, y_center_bins_repeated = hf.get_position_grid(
                signal_data.x_coordinates, signal_data.y_coordinates, self.x_bin_size, self.y_bin_size,
                environment_edges=signal_data.environment_edges)

            signal_data.add_position_binned(x_grid, y_grid)

            signal_data.add_visits(x_center_bins, y_center_bins)

            signal_data.add_position_time_spent()

            DataValidator.get_valid_timepoints(signal_data, self.min_speed_threshold, self.min_visits, self.min_time_spent)

            signal_data.add_peaks_detection(self.signal_type)
            
            signal_data.add_binned_input_signal(self.nbins_cal)

            X = signal_data.input_signal_binned.copy().reshape(-1,1)

            y = signal_data.position_binned.copy()

            y_pred = self.run_classifier(X,y,kfolds = self.num_of_folds)

            continuous_error,mean_error = self.get_continuous_error(y_pred,signal_data.x_coordinates,signal_data.y_coordinates,
                            x_center_bins_repeated,y_center_bins_repeated)

            spatial_error = self.get_spatial_error(continuous_error,y,x_center_bins,y_center_bins)
                                
            spatial_error_smoothed = self.get_smoothed_spatial_error(spatial_error,x_center_bins,y_center_bins, self.map_smoothing_sigma_x, self.map_smoothing_sigma_y)


            results = self.parallelize_surrogate(X,y,self.num_of_folds,signal_data.x_coordinates,signal_data.y_coordinates,x_center_bins, y_center_bins,
                            x_center_bins_repeated, y_center_bins_repeated, signal_data.sampling_rate, self.shift_time,
                            self.num_cores, self.num_surrogates)

            events_error = np.nanmean(continuous_error[signal_data.peaks_idx])

            mean_error_shifted = []
            spatial_error_shifted = []
            events_error_shifted = []
            for perm in range(self.num_surrogates):
                mean_error_shifted.append(results[perm][0])
                spatial_error_shifted.append(results[perm][1])
                continuous_error_shifted = results[perm][2]
                events_error_shifted.append(np.nanmean(continuous_error_shifted[signal_data.peaks_idx]))
            mean_error_shifted = np.array(mean_error_shifted)
            spatial_error_shifted = np.array(spatial_error_shifted)
            events_error_shifted = np.array(events_error_shifted)

            mean_error_zscored, mean_error_centered = info.get_mutual_information_zscored(mean_error, mean_error_shifted)
            mean_error_statistic = be.calculate_p_value(mean_error, mean_error_shifted, alternative='less')
            mean_error_pvalue = mean_error_statistic.p_value

            events_error_zscored, events_error_centered = info.get_mutual_information_zscored(events_error, events_error_shifted)
            events_error_statistic = be.calculate_p_value(events_error, events_error_shifted, alternative='less')
            events_error_pvalue = events_error_statistic.p_value


            inputdict = dict()
            
            inputdict['continuous_error'] = continuous_error
            inputdict['mean_error'] = mean_error
            inputdict['mean_error_zscored'] = mean_error_zscored
            inputdict['mean_error_centered'] = mean_error_centered
            inputdict['mean_error_shifted'] = mean_error_shifted     
            inputdict['mean_error_pvalue'] = mean_error_pvalue     
 
            inputdict['events_error'] = events_error     
            inputdict['events_error_zscored'] = events_error_zscored     
            inputdict['events_error_centered'] = events_error_centered  
            inputdict['events_error_shifted'] = events_error_shifted    
            inputdict['events_error_pvalue'] = events_error_pvalue     

            inputdict['spatial_error'] = spatial_error
            inputdict['spatial_error_shifted'] = spatial_error_shifted

            inputdict['spatial_error_smoothed'] = spatial_error_smoothed
            inputdict['x_grid'] = x_grid
            inputdict['y_grid'] = y_grid
            inputdict['x_center_bins'] = x_center_bins
            inputdict['y_center_bins'] = y_center_bins

            inputdict['input_parameters'] = self.__dict__['input_parameters']

            filename = hf.filename_constructor(self.saving_string,self.animal_id,self.dataset,self.day,self.neuron,self.trial)

            if self.saving == True:
                hf.caller_saving(inputdict,filename,self.saving_path)
            else:
                print('File not saved!')

        return inputdict
        
    

    def parallelize_surrogate(self,X,y,kfolds,x_coordinates,y_coordinates,x_center_bins, y_center_bins,
                            x_center_bins_repeated, y_center_bins_repeated, sampling_rate, shift_time,
                            num_cores, num_surrogates):
        

        with tqdm_joblib(tqdm(desc="Processing Surrogates", total=num_surrogates)) as progress_bar:
            results = Parallel(n_jobs=num_cores)(
                delayed(self.run_classifier_surrogate)
                (
                    X, y, kfolds, x_coordinates,y_coordinates, x_center_bins, y_center_bins, 
                    x_center_bins_repeated, y_center_bins_repeated, sampling_rate, shift_time
                )
                for _ in range(num_surrogates)
            )
        return results


    def run_classifier_surrogate(self, X, y, kfolds, x_coordinates,y_coordinates, x_center_bins, y_center_bins, 
                                x_center_bins_repeated, y_center_bins_repeated, sampling_rate, shift_time):
        
        X_shifted = surrogate.get_signal_surrogate(X, sampling_rate, shift_time, axis=0)
        y_pred_shifted = self.run_classifier(X_shifted, y, kfolds)

        continuous_error_shifted,mean_error_shifted = self.get_continuous_error(y_pred_shifted,x_coordinates,y_coordinates,
                                                                        x_center_bins_repeated,y_center_bins_repeated)

        spatial_error_shifted = self.get_spatial_error(continuous_error_shifted,y,x_center_bins,y_center_bins)

        return mean_error_shifted,spatial_error_shifted,continuous_error_shifted

    
    def run_classifier(self, X, y, kfolds=3):
        
        from sklearn.naive_bayes import GaussianNB

        def random_argmax(probabilities):
            """
            Selects a class index randomly among those with the highest probability.
            
            Parameters:
            probabilities (array-like): Array of class probabilities for a single sample.
            
            Returns:
            int: Index of the selected class.
            """
            max_prob = np.nanmax(probabilities)
            candidates = np.flatnonzero(probabilities == max_prob)
            return np.random.choice(candidates)


        folds_samples = decoding.kfold_split_continuous(y, kfolds)
        y_pred = []

        for test_fold in range(kfolds):
            X_train, X_test, y_train, y_test = decoding.kfold_run(X, y, folds_samples, test_fold)

            priors_in = np.ones(np.unique(y_train).shape[0]) / np.unique(y_train).shape[0]

            gnb = GaussianNB(priors=priors_in)
            gnb.fit(X_train, y_train)
            predict_probability = gnb.predict_proba(X_test)

            y_pred_fold = []
            for probs in predict_probability:
                selected_class_index = random_argmax(probs)
                y_pred_fold.append(gnb.classes_[selected_class_index])

            y_pred.append(np.array(y_pred_fold).astype(int))

        y_pred = np.concatenate(y_pred)
        return y_pred


    def get_continuous_error(self,y_pred,x_coordinates,y_coordinates,
                            x_center_bins_repeated,y_center_bins_repeated):

        diffx = (x_center_bins_repeated[y_pred]-x_coordinates)**2
        diffy = (y_center_bins_repeated[y_pred]-y_coordinates)**2

        continuous_error = np.sqrt(diffx + diffy)
        continuous_error = np.array(continuous_error)
        mean_error = np.nanmean(continuous_error)
        return continuous_error,mean_error


    def get_spatial_error(self,continuous_error,y,x_center_bins,y_center_bins):

        spatial_error = np.zeros((y_center_bins.shape[0],x_center_bins.shape[0]))*np.nan
        count = 0
        for xx in range(spatial_error.shape[1]):
            for yy in range(spatial_error.shape[0]):
                y_mask = y == count
                if np.nansum(y_mask) > 0:
                    spatial_error[yy,xx] = np.nanmean(continuous_error[y_mask])
                count += 1
            
        return spatial_error

    def get_smoothed_spatial_error(self,spatial_error,x_center_bins,y_center_bins, sigma_x,sigma_y):

        if sigma_y is None:
            sigma_y = sigma_x

        sigma_x_points = smooth.get_sigma_points(sigma_x,x_center_bins)
        sigma_y_points = smooth.get_sigma_points(sigma_y,y_center_bins)

        kernel, (x_mesh, y_mesh) = smooth.generate_2d_gaussian_kernel(sigma_x_points, sigma_y_points, radius_x=None, radius_y=None, truncate=4.0)

        spatial_error_smoothed = smooth.gaussian_smooth_2d(spatial_error, kernel, handle_nans=False)

        return spatial_error_smoothed