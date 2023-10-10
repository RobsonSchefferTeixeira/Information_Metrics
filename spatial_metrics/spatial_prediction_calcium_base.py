import numpy as np
from scipy.io import loadmat
import os
from scipy import stats as stats
from joblib import Parallel, delayed
from sklearn.naive_bayes import GaussianNB
import spatial_metrics.detect_peaks as dp
import spatial_metrics.helper_functions as hf
import warnings


class SpatialPrediction:
    def __init__(self,**kwargs):
           
        kwargs.setdefault('animal_id', None)
        kwargs.setdefault('day', None)
        kwargs.setdefault('neuron', None)
        kwargs.setdefault('trial', None)
        kwargs.setdefault('dataset', None)
        kwargs.setdefault('mean_video_srate', 30.)  
        kwargs.setdefault('min_time_spent', 0.1)  
        kwargs.setdefault('min_visits', 1)
        kwargs.setdefault('min_speed_threshold', 2.5)  
        kwargs.setdefault('x_bin_size', 1)  # in cm
        kwargs.setdefault('y_bin_size', 1)  # in cm
        kwargs.setdefault('environment_edges', []) # [[x1,x2],[y1,y2]]
        kwargs.setdefault('shift_time', 10)  # in seconds
        kwargs.setdefault('num_cores', 1)
        kwargs.setdefault('num_surrogates', 200)
        kwargs.setdefault('saving_path', os.getcwd())
        kwargs.setdefault('saving', False)
        kwargs.setdefault('saving_string', 'SpatialPrediction')
        kwargs.setdefault('num_of_folds', 10)
        kwargs.setdefault('smoothing_size', 2)        
        
        valid_kwargs = ['animal_id','day','neuron','dataset','trial', 'mean_video_srate',
                        'min_time_spent','min_visits','min_speed_threshold','smoothing_size',
                        'x_bin_size','y_bin_size','shift_time','num_cores',
                        'num_surrogates','saving_path','saving','saving_string',
                        'num_of_folds','environment_edges']
        
        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            setattr(self, k, v)
            
        self.__dict__['input_parameters'] = kwargs
        
    def main(self,calcium_imag,track_timevector,x_coordinates,y_coordinates):
        
        if np.all(np.isnan(calcium_imag)):
            warnings.warn("Signal contains only NaN's")
            inputdict = np.nan
        else:

            calcium_imag = np.atleast_2d(calcium_imag)
            if np.any(np.isnan(calcium_imag)):
                I_keep = ~np.isnan(calcium_imag)
                calcium_imag = calcium_imag[:,I_keep]
                track_timevector = track_timevector[I_keep]
                x_coordinates = x_coordinates[I_keep]
                y_coordinates = y_coordinates[I_keep]

    
            speed = hf.get_speed(x_coordinates, y_coordinates, track_timevector)

            x_grid, y_grid, x_center_bins, y_center_bins, x_center_bins_repeated, y_center_bins_repeated = \
                hf.get_position_grid(x_coordinates, y_coordinates, self.x_bin_size,
                                       self.y_bin_size, environment_edges=self.environment_edges)

            position_binned = hf.get_binned_2Dposition(x_coordinates, y_coordinates, x_grid, y_grid)

            visits_bins, new_visits_times = hf.get_visits(x_coordinates, y_coordinates, position_binned,
                                                            x_center_bins, y_center_bins)
            time_spent_inside_bins = hf.get_position_time_spent(position_binned, self.mean_video_srate)

            I_keep = self.get_valid_timepoints(calcium_imag, speed, visits_bins, time_spent_inside_bins,
                                               self.min_speed_threshold, self.min_visits, self.min_time_spent)

            calcium_imag_valid = calcium_imag[:,I_keep].copy()
            x_coordinates_valid = x_coordinates[I_keep].copy()
            y_coordinates_valid = y_coordinates[I_keep].copy()
            track_timevector_valid = track_timevector[I_keep].copy()
            visits_bins_valid = visits_bins[I_keep].copy()
            position_binned_valid = position_binned[I_keep].copy()


            all_I_peaks = []
            events_x_localization = []
            events_y_localization = []
            numb_events = []
            events_amp = []
            for cc in range(calcium_imag_valid.shape[0]):
                I_peaks = dp.detect_peaks(np.squeeze(calcium_imag_valid[cc,:]),mpd=0.5*self.mean_video_srate,mph=1.*np.nanstd(np.squeeze(calcium_imag_valid[cc,:])))
                events_x_localization.append(x_coordinates_valid[I_peaks])
                events_y_localization.append(y_coordinates_valid[I_peaks])
                all_I_peaks.append(I_peaks)
                numb_events.append(I_peaks.shape[0])
                events_amp.append(np.squeeze(calcium_imag_valid[cc,:][I_peaks]))
                
            Input_Variable,Target_Variable = self.define_input_variables(calcium_imag_valid,position_binned_valid,time_bin=1)

            concat_accuracy,concat_continuous_error,concat_mean_error,concat_continuous_accuracy = self.run_all_folds(Input_Variable,Target_Variable,x_coordinates_valid,y_coordinates_valid,self.num_of_folds,x_center_bins_repeated,y_center_bins_repeated,self.mean_video_srate)

            spatial_error = self.get_spatial_error(concat_continuous_error,Target_Variable,x_center_bins,y_center_bins)
            smoothed_spatial_error = self.smooth_spatial_error(spatial_error,spatial_bins=self.smoothing_size)

            inputdict = dict()
            inputdict['concat_accuracy'] = concat_accuracy
            inputdict['concat_continuous_error'] = concat_continuous_error
            inputdict['concat_continuous_accuracy'] = concat_continuous_accuracy
            inputdict['concat_mean_error'] = concat_mean_error     
            inputdict['spatial_error'] = spatial_error
            inputdict['spatial_error_smoothed'] = smoothed_spatial_error
            inputdict['x_grid'] = x_grid
            inputdict['y_grid'] = y_grid
            inputdict['x_center_bins'] = x_center_bins
            inputdict['y_center_bins'] = y_center_bins
            inputdict['numb_events'] = numb_events
            inputdict['events_index'] = all_I_peaks
            inputdict['events_amp'] = events_amp
            inputdict['events_x_localization'] = events_x_localization
            inputdict['events_y_localization'] = events_y_localization
            inputdict['input_parameters'] = self.__dict__['input_parameters']

            filename = hf.filename_constructor(self.saving_string,self.animal_id,self.dataset,self.day,self.neuron,self.trial)

            if self.saving == True:
                hf.caller_saving(inputdict,filename,self.saving_path)
            else:
                print('File not saved!')

        return inputdict
    
    def define_input_variables(self,calcium_imag_valid,position_binned_valid,time_bin):

        calcium_imag_valid_norm = hf.min_max_norm(calcium_imag_valid)
        Input_Variable = np.copy(calcium_imag_valid_norm.T)
        Target_Variable = np.copy(position_binned_valid)
        return Input_Variable,Target_Variable           

    
    def run_all_folds(self,Input_Variable,Target_Variable,x_coordinates_valid,y_coordinates_valid,num_of_folds,x_center_bins_repeated,y_center_bins_repeated,mean_video_srate):

        concat_continuous_accuracy = []
        concat_continuous_error = []
        concat_mean_error = []
        concat_accuracy = []
        for fold in range(1,num_of_folds+1):
            X_train,y_train,X_test,y_test,Trials_training_set,Trials_testing_set = self.get_fold_trials(Input_Variable,Target_Variable,fold,num_of_folds)
            classifier_accuracy,y_pred,predict_proba = self.run_classifier(X_train,y_train,X_test,y_test)
            
            concat_continuous_accuracy.append((y_pred == y_test).astype(int))
            
            x_coordinates_test = x_coordinates_valid[Trials_testing_set].copy()
            y_coordinates_test = y_coordinates_valid[Trials_testing_set].copy()

            continuous_error,mean_error = self.get_continuous_error(y_test,y_pred,x_coordinates_test,y_coordinates_test,x_center_bins_repeated,y_center_bins_repeated)

            concat_accuracy.append(classifier_accuracy)
            concat_continuous_error.append(continuous_error)
            concat_mean_error.append(mean_error)

        concat_continuous_accuracy = np.concatenate(concat_continuous_accuracy)
        concat_accuracy = np.array(concat_accuracy)
        concat_continuous_error = np.concatenate(concat_continuous_error)
        concat_mean_error = np.array(concat_mean_error)

        return concat_accuracy,concat_continuous_error,concat_mean_error,concat_continuous_accuracy

    def get_fold_trials(self,Input_Variable,Target_Variable,fold,num_of_folds):

        window_size = int(np.floor(Input_Variable.shape[0]/num_of_folds))
        I_start = np.arange(0,Input_Variable.shape[0],window_size)

        if fold==(num_of_folds):
            Trials_testing_set =  np.arange(I_start[fold-1],Input_Variable.shape[0]).astype(int)
        elif (fold == 0) | (fold > num_of_folds):
            raise ValueError('Fold number must be a integer between 1 and num_of_folds') 
        else:
            Trials_testing_set =  np.arange(I_start[fold-1],I_start[fold]).astype(int)

        Trials_training_set = np.setdiff1d(range(Input_Variable.shape[0]),Trials_testing_set)

        X_train = Input_Variable[Trials_training_set,:].copy()
        y_train = Target_Variable[Trials_training_set].copy()

        X_test = Input_Variable[Trials_testing_set,:].copy()
        y_test = Target_Variable[Trials_testing_set].copy()

        return X_train,y_train,X_test,y_test,Trials_training_set,Trials_testing_set

    def run_classifier(self,X_train,y_train,X_test,y_test):

        priors_in = np.ones(np.unique(y_train).shape[0])/np.unique(y_train).shape[0]

        gnb = GaussianNB(priors = priors_in)
        gnb.fit(X_train, y_train)
        predict_proba = gnb.predict_proba(X_test)

        accuracy_original = gnb.score(X_test, y_test)

        # When the classifier assigns the same probability to all classes, the argmax returns the lowest index.
        # To avoid this, we add a small random number to the probabilities.
        # y_pred = gnb.predict(X_test)
        # y_test = y_test.astype(int)
        # y_pred = y_pred.astype(int)
        y_pred = []
        for ii in range(len(predict_proba)):
            predict_proba[ii] += np.random.randn(predict_proba[ii].shape[0]) / 100
            y_pred.append(gnb.classes_[np.argmax(predict_proba[ii])])
        y_pred = np.array(y_pred).astype(int)

        return accuracy_original,y_pred,predict_proba


    
    def get_spatial_error(self,continuous_error,y_test,x_center_bins,y_center_bins):

        pred_dist_grid_original = np.zeros((y_center_bins.shape[0],x_center_bins.shape[0]))*np.nan
        count = 0
        for xx in range(pred_dist_grid_original.shape[1]):
            for yy in range(pred_dist_grid_original.shape[0]):

                I_test = np.where(y_test == count)[0]
                if len(I_test)>0:
                    pred_dist_grid_original[yy,xx] = np.nanmean(continuous_error[I_test])

                count += 1
        return pred_dist_grid_original


    def get_continuous_error(self,y_test,y_pred,x_coordinates,y_coordinates,
                             x_center_bins_repeated,y_center_bins_repeated):

        diffx = (x_center_bins_repeated[y_pred]-x_coordinates)**2
        diffy = (y_center_bins_repeated[y_pred]-y_coordinates)**2

        continuous_nearest_dist_to_predicted = np.sqrt(diffx + diffy)
        continuous_nearest_dist_to_predicted = np.array(continuous_nearest_dist_to_predicted)
        mean_nearest_dist_to_predicted = np.nanmean(continuous_nearest_dist_to_predicted)
        return continuous_nearest_dist_to_predicted,mean_nearest_dist_to_predicted


    def get_valid_timepoints(self, calcium_imag, speed, visits_bins, time_spent_inside_bins, min_speed_threshold,
                             min_visits, min_time_spent):

        # min speed
        I_speed_thres = speed >= min_speed_threshold

        # min visits
        I_visits_times_thres = visits_bins >= min_visits

        # min time spent
        I_time_spent_thres = time_spent_inside_bins >= min_time_spent

        # valid calcium points
        I_valid_calcium = ~np.any(np.isnan(calcium_imag),0)
        # I_valid_calcium = ~np.isnan(calcium_imag)

        I_keep = I_speed_thres * I_visits_times_thres * I_time_spent_thres * I_valid_calcium
        return I_keep


    def smooth_spatial_error(self,original_spatial_error,spatial_bins=2):
        

        original_spatial_error_to_smooth = np.copy(original_spatial_error)
        I_nan = np.isnan(original_spatial_error_to_smooth)
        original_spatial_error_to_smooth[I_nan] = 0 
        smoothed_spatial_error = -hf.gaussian_smooth_2d(-original_spatial_error_to_smooth,spatial_bins)
        # smoothed_spatial_error[I_nan] = np.nan
        return smoothed_spatial_error



class SpatialPredictionSurrogates(SpatialPrediction):
    
    def main(self,calcium_imag,track_timevector,x_coordinates,y_coordinates):
        
       
        if np.all(np.isnan(calcium_imag)):
            warnings.warn("Signal contains only NaN's")
            inputdict = np.nan
        else:

            calcium_imag = np.atleast_2d(calcium_imag)
            if np.any(np.isnan(calcium_imag)):
                I_keep = ~np.isnan(calcium_imag)
                calcium_imag = calcium_imag[:,I_keep]
                track_timevector = track_timevector[I_keep]
                x_coordinates = x_coordinates[I_keep]
                y_coordinates = y_coordinates[I_keep]

    
            speed = hf.get_speed(x_coordinates, y_coordinates, track_timevector)

            x_grid, y_grid, x_center_bins, y_center_bins, x_center_bins_repeated, y_center_bins_repeated = hf.get_position_grid(x_coordinates, y_coordinates, self.x_bin_size,self.y_bin_size, environment_edges=self.environment_edges)

            position_binned = hf.get_binned_2Dposition(x_coordinates, y_coordinates, x_grid, y_grid)

            visits_bins, new_visits_times = hf.get_visits(x_coordinates, y_coordinates, position_binned,
                                                            x_center_bins, y_center_bins)
            time_spent_inside_bins = hf.get_position_time_spent(position_binned, self.mean_video_srate)

            I_keep = self.get_valid_timepoints(calcium_imag, speed, visits_bins, time_spent_inside_bins,
                                               self.min_speed_threshold, self.min_visits, self.min_time_spent)

            calcium_imag_valid = calcium_imag[:,I_keep].copy()
            x_coordinates_valid = x_coordinates[I_keep].copy()
            y_coordinates_valid = y_coordinates[I_keep].copy()
            track_timevector_valid = track_timevector[I_keep].copy()
            visits_bins_valid = visits_bins[I_keep].copy()
            position_binned_valid = position_binned[I_keep].copy()


            results = self.parallelize_surrogate(calcium_imag,I_keep, position_binned_valid,x_coordinates_valid,y_coordinates_valid,self.num_of_folds, x_center_bins,y_center_bins,self.x_bin_size,self.y_bin_size,x_center_bins_repeated,y_center_bins_repeated,self.mean_video_srate,self.num_cores,self.num_surrogates,self.shift_time)

            concat_accuracy = []
            concat_continuous_error = []
            concat_mean_error = []
            I_peaks = []
            spatial_error = []
            smoothed_spatial_error = []
            numb_events = []
            events_amp = []
            events_x_localization = []
            events_y_localization = []
            spatial_error = []
            smoothed_spatial_error = []
            all_I_peaks = []
            events_x_localization = []
            events_y_localization = []
            numb_events = []
            events_amp = []
            concat_continuous_accuracy = []
            for surr in range(self.num_surrogates):
                concat_accuracy.append(results[surr][0])
                concat_continuous_error.append(results[surr][1])
                concat_mean_error.append(results[surr][2])
                spatial_error.append(results[surr][3])
                smoothed_spatial_error.append(results[surr][4])
                all_I_peaks.append(results[surr][5])
                events_x_localization.append(results[surr][6])
                events_y_localization.append(results[surr][7])
                numb_events.append(results[surr][8])
                events_amp.append(results[surr][9])
                concat_continuous_accuracy.append(results[surr][10])

            inputdict = dict()
            inputdict['concat_accuracy'] = concat_accuracy
            inputdict['concat_continuous_error'] = concat_continuous_error
            inputdict['concat_continuous_accuracy'] = concat_continuous_accuracy
            inputdict['concat_mean_error'] = concat_mean_error
            inputdict['spatial_error'] = spatial_error
            inputdict['spatial_error_smoothed'] = smoothed_spatial_error
            inputdict['x_grid'] = x_grid
            inputdict['y_grid'] = y_grid
            inputdict['x_center_bins'] = x_center_bins
            inputdict['y_center_bins'] = y_center_bins
            inputdict['numb_events'] = numb_events
            inputdict['events_index'] = all_I_peaks
            inputdict['events_amp'] = events_amp
            inputdict['events_x_localization'] = events_x_localization
            inputdict['events_y_localization'] = events_y_localization
            inputdict['input_parameters'] = self.__dict__['input_parameters']

            filename = hf.filename_constructor(self.saving_string,self.animal_id,self.dataset,self.day,self.neuron,self.trial)

            if self.saving == True:
                hf.caller_saving(inputdict,filename,self.saving_path)
            else:
                print('File not saved!')

        return inputdict
    
            
 

    def parallelize_surrogate(self,calcium_imag,I_keep,position_binned_valid,x_coordinates_valid,y_coordinates_valid,num_of_folds,x_center_bins,y_center_bins,
        x_bin_size,y_bin_size,x_center_bins_repeated,y_center_bins_repeated,mean_video_srate,num_cores,num_surrogates,shift_time):
        
        results = Parallel(n_jobs=num_cores)(delayed(self.shuffle_and_run_all_folds)(calcium_imag,I_keep,position_binned_valid,x_coordinates_valid,y_coordinates_valid,num_of_folds,x_center_bins,y_center_bins,
        x_bin_size,y_bin_size,x_center_bins_repeated,y_center_bins_repeated,mean_video_srate,shift_time) for permi in range(num_surrogates))
        
        return results
    
    
    
    def shuffle_and_run_all_folds(self,calcium_imag,I_keep,position_binned_valid,x_coordinates_valid,y_coordinates_valid,num_of_folds, x_center_bins,y_center_bins,x_bin_size,y_bin_size,x_center_bins_repeated,y_center_bins_repeated,mean_video_srate,shift_time):
             
        calcium_imag_shuffled = self.get_surrogate(calcium_imag,mean_video_srate,shift_time)
        calcium_imag_shuffled_valid = calcium_imag_shuffled[:,I_keep].copy()
        
        Input_Variable_Shuffled,Target_Variable = self.define_input_variables(calcium_imag_shuffled_valid,position_binned_valid,time_bin=1)
        
        concat_accuracy,concat_continuous_error,concat_mean_error,concat_continuous_accuracy = self.run_all_folds(Input_Variable_Shuffled,                           Target_Variable,x_coordinates_valid,y_coordinates_valid,self.num_of_folds,x_center_bins_repeated,                                                     y_center_bins_repeated,self.mean_video_srate)
        
        spatial_error = self.get_spatial_error(concat_continuous_error,Target_Variable,x_center_bins,y_center_bins)
        smoothed_spatial_error = self.smooth_spatial_error(spatial_error,spatial_bins=2)

        all_I_peaks = []
        events_x_localization = []
        events_y_localization = []
        numb_events = []
        events_amp = []
        for cc in range(calcium_imag_shuffled_valid.shape[0]):
            I_peaks = dp.detect_peaks(np.squeeze(calcium_imag_shuffled_valid[cc,:]),mpd=0.5*self.mean_video_srate,mph=1.*np.nanstd(np.squeeze(calcium_imag_shuffled_valid[cc,:])))

            events_x_localization.append(x_coordinates_valid[I_peaks])
            events_y_localization.append(y_coordinates_valid[I_peaks])
            all_I_peaks.append(I_peaks)
            numb_events.append(I_peaks.shape[0])
            events_amp.append(np.squeeze(calcium_imag_shuffled_valid[cc,:][I_peaks]))

        return concat_accuracy,concat_continuous_error,concat_mean_error,spatial_error,smoothed_spatial_error,all_I_peaks,events_x_localization,events_y_localization,numb_events,events_amp,concat_continuous_accuracy

    def get_surrogate(self,input_vector,mean_video_srate,shift_time):
        input_vector = np.atleast_2d(input_vector)
        I_break = np.random.choice(np.arange(-shift_time*mean_video_srate,mean_video_srate*shift_time),1)[0].astype(int)
        input_vector_shuffled = np.concatenate([input_vector[:,I_break:], input_vector[:,0:I_break]],1)
        return input_vector_shuffled