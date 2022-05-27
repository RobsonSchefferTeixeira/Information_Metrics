import numpy as np
from scipy.io import loadmat
import os
from scipy import stats as stats
from joblib import Parallel, delayed
from sklearn.naive_bayes import GaussianNB
import detect_peaks as dp
import spatial_metrics.helper_functions as hf

class SpatialPrediction:
    def __init__(self,**kwargs):
           
        kwargs.setdefault('session', [])  
        kwargs.setdefault('day', [])  
        kwargs.setdefault('ch', []) 
        kwargs.setdefault('trial', 0) 
        kwargs.setdefault('dataset', [])  
        kwargs.setdefault('mean_video_srate', 30.)  
        kwargs.setdefault('mintimespent', 0.1)  
        kwargs.setdefault('minvisits', 1)  
        kwargs.setdefault('speed_threshold', 2.5)  
        kwargs.setdefault('x_bin_size', 1)  # in cm
        kwargs.setdefault('y_bin_size', 1)  # in cm
        kwargs.setdefault('environment_edges', []) # [[x1,x2],[y1,y2]]
        kwargs.setdefault('shuffling_shift', 10)  # in seconds
        kwargs.setdefault('num_cores', 1)  
        kwargs.setdefault('num_surrogates', 200)          
        kwargs.setdefault('saving_path', os.getcwd())  
        kwargs.setdefault('saving', False)  
        kwargs.setdefault('saving_string', [])             
        kwargs.setdefault('num_of_folds', 10) 
        
        
        valid_kwargs = ['session','day','ch','dataset','trial', 'mean_video_srate',
                        'mintimespent','minvisits','speed_threshold',
                        'x_bin_size','y_bin_size','shuffling_shift','num_cores',
                        'num_surrogates','saving_path','saving','saving_string',
                        'num_of_folds','environment_edges']
        
        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            setattr(self, k, v)
            
        self.__dict__['input_parameters'] = kwargs
        
    def main(self,calcium_signal,track_timevector,x_coordinates,y_coordinates):
        
        x_grid,y_grid,x_center_bins,y_center_bins,x_center_bins_repeated,y_center_bins_repeated = self.get_position_grid(x_coordinates,y_coordinates,self.x_bin_size,self.y_bin_size,environment_edges=self.environment_edges)

        position_binned = self.get_binned_2Dposition(x_coordinates,y_coordinates,x_grid,y_grid)

        Input_Variable,Target_Variable,x_coordinates_valid,y_coordinates_valid,I_valid = self.get_valid_timepoints(calcium_signal,position_binned,x_coordinates,y_coordinates)

                    
        concat_accuracy,concat_continuous_error,concat_mean_error_classic, concat_continuous_error_center_of_mass,concat_mean_error_center_of_mass, I_peaks = self.run_all_folds(Input_Variable,Target_Variable,x_coordinates_valid,y_coordinates_valid,self.num_of_folds, x_center_bins,y_center_bins,x_center_bins_repeated,y_center_bins_repeated,self.mean_video_srate)
        

        spatial_error_classic = self.get_spatial_error(concat_continuous_error,Target_Variable,x_center_bins,y_center_bins)
        smoothed_spatial_error_classic = self.smooth_spatial_error(spatial_error_classic,spatial_bins=2)

        spatial_error_center_of_mass = self.get_spatial_error(concat_continuous_error_center_of_mass,Target_Variable,x_center_bins,y_center_bins)
        smoothed_spatial_center_of_mass = self.smooth_spatial_error(spatial_error_center_of_mass,spatial_bins=2)

        
    
        inputdict = dict()
        inputdict['concat_accuracy'] = concat_accuracy
        inputdict['concat_continuous_error'] = concat_continuous_error
        inputdict['concat_mean_error_classic'] = concat_mean_error_classic        
        inputdict['spatial_error_classic'] = spatial_error_classic
        inputdict['smoothed_spatial_error_classic'] = smoothed_spatial_error_classic
        inputdict['concat_continuous_error_center_of_mass'] = concat_continuous_error_center_of_mass
        inputdict['concat_mean_error_center_of_mass'] = concat_mean_error_center_of_mass
        inputdict['spatial_error_center_of_mass'] = spatial_error_center_of_mass
        inputdict['smoothed_spatial_center_of_mass'] = smoothed_spatial_center_of_mass
        inputdict['x_grid'] = x_grid
        inputdict['y_grid'] = y_grid
        inputdict['x_center_bins'] = x_center_bins
        inputdict['y_center_bins'] = y_center_bins
        inputdict['numb_events'] = I_peaks.shape[0]
        inputdict['events_index'] = I_peaks
        inputdict['events_amp'] = Input_Variable[I_peaks]
        inputdict['events_x_localization'] = x_coordinates_valid[I_peaks]
        inputdict['events_y_localization'] = y_coordinates_valid[I_peaks]
        
        

        
        
        if self.saving == True:
            if self.trial == 0:
                filename = self.session + '.' + self.saving_string + '.SpatialPrediction.Original.' + self.dataset + '.Day.' + str(self.day) + '.Ch.' + str(self.ch)
                self.caller_saving(inputdict,filename,self.saving_path)

                filename = self.session + '.' + self.saving_string + '.SpatialPrediction.Parameters.' + self.dataset + '.Day.' + str(self.day) + '.Ch.' + str(self.ch)
                self.caller_saving(self.__dict__['input_parameters'],filename,self.saving_path)
            else:
                filename = self.session + '.' + self.saving_string + '.SpatialPrediction.Original.' + self.dataset + '.Day.' + str(self.day) + '.Ch.' + str(self.ch) + '.Trial.' + str(self.trial)
                self.caller_saving(inputdict,filename,self.saving_path)

                filename = self.session + '.' + self.saving_string + '.SpatialPrediction.Parameters.' + self.dataset + '.Day.' + str(self.day) + '.Ch.' + str(self.ch) + '.Trial.' + str(self.trial)
                self.caller_saving(self.__dict__['input_parameters'],filename,self.saving_path)
                
                
                
        else:
            print('File not saved!')
        
        
        return inputdict
    
            
    def caller_saving(self,inputdict,filename,saving_path):
        print('Saving data file...')
        os.chdir(saving_path)
        output = open(filename, 'wb') 
        np.save(output,inputdict)
        output.close()
     
  
    
    def run_all_folds(self,Input_Variable,Target_Variable,x_coordinates_valid,y_coordinates_valid,num_of_folds,x_center_bins,y_center_bins,x_center_bins_repeated,y_center_bins_repeated,mean_video_srate):

        concat_continuous_error = []
        concat_mean_error_classic = []
        # concat_pred_dist_grid_classic = []

        concat_continuous_error_center_of_mass = []
        concat_mean_error_center_of_mass = []
        # concat_pred_dist_grid_center_of_mass = []

        concat_accuracy = []

        I_peaks = dp.detect_peaks(np.squeeze(Input_Variable),mpd=0.5*mean_video_srate,mph=1.*np.nanstd(np.squeeze(Input_Variable)))

        for fold in range(1,num_of_folds+1):


            X_train,y_train,X_test,y_test,Trials_training_set,Trials_testing_set = self.get_fold_trials(Input_Variable,Target_Variable,
                                                                                                      fold,num_of_folds)

            classifier_accuracy,y_pred,predict_proba = self.run_classifier(X_train,y_train,X_test,y_test)

            x_coordinates_test = x_coordinates_valid[Trials_testing_set].copy()
            y_coordinates_test = y_coordinates_valid[Trials_testing_set].copy()

            continuous_error_classic,mean_error_classic = self.get_classic_continuous_error(y_test,y_pred,x_coordinates_test,y_coordinates_test,
                                                                                          x_center_bins_repeated,y_center_bins_repeated)

            # pred_dist_grid_classic = self.get_spatial_error(continuous_error_classic,y_test,x_center_bins,y_center_bins)

            continuous_error_center_of_mass,mean_error_center_of_mass = self.get_center_of_mass_continuous_error(y_train,predict_proba,                                                                                   x_coordinates_test,y_coordinates_test,x_center_bins,y_center_bins,
                                                                      x_center_bins_repeated,y_center_bins_repeated)

            # pred_dist_grid_center_of_mass = self.get_spatial_error(continuous_error_center_of_mass,y_test,x_center_bins,y_center_bins)


            concat_accuracy.append(classifier_accuracy)

            concat_continuous_error.append(continuous_error_classic)
            concat_mean_error_classic.append(mean_error_classic)
            # concat_pred_dist_grid_classic.append(pred_dist_grid_classic)

            concat_continuous_error_center_of_mass.append(continuous_error_center_of_mass)
            concat_mean_error_center_of_mass.append(mean_error_center_of_mass)     
            # concat_pred_dist_grid_center_of_mass.append(pred_dist_grid_center_of_mass)


        concat_accuracy = np.array(concat_accuracy)

        concat_continuous_error = np.concatenate(concat_continuous_error)
        concat_mean_error_classic = np.array(concat_mean_error_classic)
        # concat_pred_dist_grid_classic = np.array(concat_pred_dist_grid_classic)

        concat_continuous_error_center_of_mass = np.concatenate(concat_continuous_error_center_of_mass)
        concat_mean_error_center_of_mass = np.array(concat_mean_error_center_of_mass)
        # concat_pred_dist_grid_center_of_mass = np.array(concat_pred_dist_grid_center_of_mass)




        return concat_accuracy,concat_continuous_error,concat_mean_error_classic,                                                                                    concat_continuous_error_center_of_mass,concat_mean_error_center_of_mass,I_peaks

    def get_position_grid(self,x_coordinates,y_coordinates,x_bin_size,y_bin_size,**kwargs):

        # x_bin_size and y_bin_size in cm
        # environment_edges = [[x1, x2], [y1, y2]]
    
        environment_edges = kwargs.get('environment_edges')

        if environment_edges:

            x_grid = np.arange(environment_edges[0][0],environment_edges[0][1] + x_bin_size/2,x_bin_size)

            y_grid = np.arange(environment_edges[1][0],environment_edges[1][1] + y_bin_size/2,y_bin_size)

            x_center_bins = x_grid[0:-1] + x_bin_size/2
            y_center_bins = y_grid[0:-1] + y_bin_size/2

            x_center_bins_repeated = np.repeat(x_center_bins,y_center_bins.shape[0])
            y_center_bins_repeated = np.tile(y_center_bins,x_center_bins.shape[0])

        else:     

            x_grid = np.arange(np.nanmin(x_coordinates),np.nanmax(x_coordinates) + x_bin_size/2,x_bin_size)
            y_grid = np.arange(np.nanmin(y_coordinates),np.nanmax(y_coordinates) + y_bin_size/2,y_bin_size)

            x_center_bins = x_grid[0:-1] + x_bin_size/2
            y_center_bins = y_grid[0:-1] + y_bin_size/2


            x_center_bins_repeated = np.repeat(x_center_bins,y_center_bins.shape[0])
            y_center_bins_repeated = np.tile(y_center_bins,x_center_bins.shape[0])


        return x_grid,y_grid,x_center_bins,y_center_bins,x_center_bins_repeated,y_center_bins_repeated



    def get_fold_trials(self,Input_Variable,Target_Variable,fold,num_of_folds):


        window_size = int(np.floor(Input_Variable.shape[0]/num_of_folds))
        I_start = np.arange(0,Input_Variable.shape[0],window_size)

        if fold==(num_of_folds):
            Trials_testing_set =  np.arange(I_start[fold-1],Input_Variable.shape[0]).astype(int)
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

        y_pred = gnb.predict(X_test)
        y_test = y_test.astype(int)
        y_pred = y_pred.astype(int)

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




    def get_classic_continuous_error(self,y_test,y_pred,x_coordinates,y_coordinates,x_center_bins_repeated,y_center_bins_repeated):

        diffx = (x_center_bins_repeated[y_pred]-x_coordinates)**2
        diffy = (y_center_bins_repeated[y_pred]-y_coordinates)**2

        continuous_nearest_dist_to_predicted = np.sqrt(diffx + diffy)
        continuous_nearest_dist_to_predicted = np.array(continuous_nearest_dist_to_predicted)
        mean_nearest_dist_to_predicted = np.nanmean(continuous_nearest_dist_to_predicted)
        return continuous_nearest_dist_to_predicted,mean_nearest_dist_to_predicted


    def get_center_of_mass_continuous_error(self,y_train,predict_proba,x_coordinates,y_coordinates,x_center_bins,                                         y_center_bins,x_center_bins_repeated,y_center_bins_repeated):


        _classes = np.sort(np.unique(y_train)).astype(int)
        continuous_nearest_dist_to_predicted = []
        for fr in range(predict_proba.shape[0]):
            predic_distr_singleTime = predict_proba[fr,:]

            prob_grid = np.zeros((x_center_bins.shape[0])*(y_center_bins.shape[0]))*np.nan
            prob_grid[_classes] = predic_distr_singleTime

            mass_x_coord = np.nansum(prob_grid*x_center_bins_repeated)
            mass_y_coord = np.nansum(prob_grid*y_center_bins_repeated)

            error_distance = np.sqrt((mass_x_coord - x_coordinates[fr])**2 +  
                                     (mass_y_coord - y_coordinates[fr])**2)

            continuous_nearest_dist_to_predicted.append(error_distance)
        continuous_nearest_dist_to_predicted = np.array(continuous_nearest_dist_to_predicted)
        mean_nearest_dist_to_predicted = np.nanmean(continuous_nearest_dist_to_predicted)

        return continuous_nearest_dist_to_predicted,mean_nearest_dist_to_predicted




    def get_binned_2Dposition(self,x_coordinates,y_coordinates,x_grid,y_grid):

        # calculate position occupancy
        position_binned = np.zeros(x_coordinates.shape)*np.nan
        count = 0
        for xx in range(0,x_grid.shape[0]-1):
            for yy in range(0,y_grid.shape[0]-1):
                check_x_ocuppancy = np.logical_and(x_coordinates >= x_grid[xx],x_coordinates < (x_grid[xx+1]))
                check_y_ocuppancy = np.logical_and(y_coordinates >= y_grid[yy],y_coordinates < (y_grid[yy+1]))
                position_binned[np.logical_and(check_x_ocuppancy,check_y_ocuppancy)] = count
                count += 1

        return position_binned

    def get_valid_timepoints(self,Input_Variable,Target_Variable,x_coordinates,y_coordinates):

        I_keep1 = ~np.isnan(Input_Variable)
        I_keep2 = ~np.isnan(Target_Variable)
        I_keep = I_keep1 & I_keep2

        Input_Variable_reshaped = Input_Variable.reshape(Input_Variable.shape[0],-1)

        return Input_Variable_reshaped[I_keep,],Target_Variable[I_keep],x_coordinates[I_keep],y_coordinates[I_keep],I_keep


    def smooth_spatial_error(self,original_spatial_error,spatial_bins=2):
        

        original_spatial_error_to_smooth = np.copy(original_spatial_error)
        I_nan = np.isnan(original_spatial_error_to_smooth)
        original_spatial_error_to_smooth[I_nan] = 0 
        smoothed_spatial_error = hf.gaussian_smooth_2d(original_spatial_error_to_smooth,spatial_bins)
        smoothed_spatial_error[I_nan] = np.nan
        return smoothed_spatial_error







class SpatialPredictionSurrogates(SpatialPrediction):
    
    def main(self,calcium_signal,track_timevector,x_coordinates,y_coordinates):
        
        x_grid,y_grid,x_center_bins,y_center_bins,x_center_bins_repeated,y_center_bins_repeated = self.get_position_grid(x_coordinates,y_coordinates,self.x_bin_size,self.y_bin_size,environment_edges=self.environment_edges)

        position_binned = self.get_binned_2Dposition(x_coordinates,y_coordinates,x_grid,y_grid)

        Input_Variable,Target_Variable,x_coordinates_valid,y_coordinates_valid,I_valid = self.get_valid_timepoints(calcium_signal,position_binned,x_coordinates,y_coordinates)

        
        results = self.parallelize_surrogate(Input_Variable,Target_Variable,x_coordinates_valid,y_coordinates_valid,self.num_of_folds, x_center_bins,y_center_bins,self.x_bin_size,self.y_bin_size,x_center_bins_repeated,y_center_bins_repeated,self.mean_video_srate,self.num_cores,self.num_surrogates,self.shuffling_shift)
        
        concat_accuracy = []
        concat_continuous_error = []
        concat_mean_error_classic = []
        concat_continuous_error_center_of_mass = []
        concat_mean_error_center_of_mass = []
        I_peaks = []
        spatial_error_classic = []
        smoothed_spatial_error_classic = []
        spatial_error_center_of_mass = []
        smoothed_spatial_center_of_mass = []
        numb_events = []
        events_amp = []
        events_x_localization = []
        events_y_localization = []    

        for surr in range(self.num_surrogates):
            concat_accuracy.append(results[surr][0])
            concat_continuous_error.append(results[surr][1])
            concat_mean_error_classic.append(results[surr][2])
            concat_continuous_error_center_of_mass.append(results[surr][3])
            concat_mean_error_center_of_mass.append(results[surr][4])
            
            I_peaks.append(results[surr][5])
            numb_events.append(results[surr][5].shape[0])
            events_x_localization.append(x_coordinates_valid[results[surr][5]])
            events_y_localization.append(y_coordinates_valid[results[surr][5]])
            
            spatial_error_classic.append(results[surr][6])
            smoothed_spatial_error_classic.append(results[surr][7])
            spatial_error_center_of_mass.append(results[surr][8])
            smoothed_spatial_center_of_mass.append(results[surr][9])
            events_amp.append(results[surr][10])
            
 
    
        inputdict = dict()
        inputdict['concat_accuracy'] = concat_accuracy
        inputdict['concat_continuous_error'] = concat_continuous_error
        inputdict['concat_mean_error_classic'] = concat_mean_error_classic        
        inputdict['spatial_error_classic'] = spatial_error_classic
        inputdict['smoothed_spatial_error_classic'] = smoothed_spatial_error_classic
        inputdict['concat_continuous_error_center_of_mass'] = concat_continuous_error_center_of_mass
        inputdict['concat_mean_error_center_of_mass'] = concat_mean_error_center_of_mass
        inputdict['spatial_error_center_of_mass'] = spatial_error_center_of_mass
        inputdict['smoothed_spatial_center_of_mass'] = smoothed_spatial_center_of_mass
        inputdict['numb_events'] = numb_events
        inputdict['events_index'] = I_peaks
        inputdict['events_amp'] = events_amp
        inputdict['events_x_localization'] = events_x_localization
        inputdict['events_y_localization'] = events_y_localization
        inputdict['x_grid'] = x_grid
        inputdict['y_grid'] = y_grid
        inputdict['x_center_bins'] = x_center_bins
        inputdict['y_center_bins'] = y_center_bins
        
        if self.saving == True:
            if self.trial == 0:
                filename = self.session + '.' + self.saving_string + '.SpatialPrediction.Surrogates.' + self.dataset + '.Day.' + str(self.day) + '.Ch.' + str(self.ch)
                self.caller_saving(inputdict,filename,self.saving_path)

            else:
                filename = self.session + '.' + self.saving_string + '.SpatialPrediction.Surrogates.' + self.dataset + '.Day.' + str(self.day) + '.Ch.' + str(self.ch) + '.Trial.' + str(self.trial)
                self.caller_saving(inputdict,filename,self.saving_path)
                
                
        else:
            print('File not saved!')
        
        
        return inputdict
    
    

    def parallelize_surrogate(self,Input_Variable,Target_Variable,x_coordinates_valid,y_coordinates_valid,num_of_folds,x_center_bins,y_center_bins,
        x_bin_size,y_bin_size,x_center_bins_repeated,y_center_bins_repeated,mean_video_srate,num_cores,num_surrogates,shuffling_shift):
        
        results = Parallel(n_jobs=num_cores)(delayed(self.shuffle_and_run_all_folds)                                                                     (Input_Variable,Target_Variable,x_coordinates_valid,y_coordinates_valid,num_of_folds,x_center_bins,y_center_bins,
        x_bin_size,y_bin_size,x_center_bins_repeated,y_center_bins_repeated,mean_video_srate,shuffling_shift) for permi in range(num_surrogates))
        
        return results
    
    
    
    def shuffle_and_run_all_folds(self,Input_Variable,Target_Variable,x_coordinates_valid,y_coordinates_valid,num_of_folds, x_center_bins,y_center_bins,x_bin_size,y_bin_size,x_center_bins_repeated,y_center_bins_repeated,mean_video_srate,shuffling_shift):
        
        Input_Variable_Shuffled = self.get_surrogate(Input_Variable,mean_video_srate,shuffling_shift)
       
                    
        concat_accuracy,concat_continuous_error,concat_mean_error_classic, concat_continuous_error_center_of_mass,concat_mean_error_center_of_mass, I_peaks = self.run_all_folds(Input_Variable_Shuffled,Target_Variable,x_coordinates_valid,y_coordinates_valid,self.num_of_folds, x_center_bins,y_center_bins,x_center_bins_repeated,y_center_bins_repeated,self.mean_video_srate)
        
        
        
        spatial_error_classic = self.get_spatial_error(concat_continuous_error,Target_Variable,x_center_bins,y_center_bins)
        smoothed_spatial_error_classic = self.smooth_spatial_error(spatial_error_classic,spatial_bins=2)

        spatial_error_center_of_mass = self.get_spatial_error(concat_continuous_error_center_of_mass,Target_Variable,x_center_bins,y_center_bins)
        smoothed_spatial_center_of_mass = self.smooth_spatial_error(spatial_error_center_of_mass,spatial_bins=2)

        events_amp = Input_Variable_Shuffled[I_peaks]
        
        return concat_accuracy,concat_continuous_error,concat_mean_error_classic, concat_continuous_error_center_of_mass,concat_mean_error_center_of_mass, I_peaks,spatial_error_classic,smoothed_spatial_error_classic,spatial_error_center_of_mass,smoothed_spatial_center_of_mass,events_amp
        
        

    def get_surrogate(self,input_vector,mean_video_srate,shuffling_shift):
            eps = np.finfo(float).eps
            I_break = np.random.choice(np.arange(-shuffling_shift*mean_video_srate,mean_video_srate*shuffling_shift),1)[0].astype(int)
            input_vector_shuffled = np.concatenate([input_vector[I_break:], input_vector[0:I_break]])

            return input_vector_shuffled