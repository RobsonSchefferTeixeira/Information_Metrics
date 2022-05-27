import numpy as np
from scipy.io import loadmat
import pandas as pd
import os
from scipy import stats as stats
import helper_functions as hf
import detect_peaks as dp
from joblib import Parallel, delayed


        
class PlaceCell:
    def __init__(self,**kwargs):
           
        kwargs.setdefault('Session', [])  
        kwargs.setdefault('day', [])  
        kwargs.setdefault('ch', [])  
        kwargs.setdefault('dataset', [])  
        kwargs.setdefault('mean_video_srate', 30.)  
        kwargs.setdefault('mintimespent', 0.1)  
        kwargs.setdefault('minvisits', 1)  
        kwargs.setdefault('speed_threshold', 2.5)  
        kwargs.setdefault('nbins_pos_x', 10)  
        kwargs.setdefault('nbins_pos_y', 10)  
        kwargs.setdefault('nbins_cal', 10)  
        kwargs.setdefault('placefield_nbins_pos_x', 50)  
        kwargs.setdefault('placefield_nbins_pos_y', 50)  
        kwargs.setdefault('num_cores', 1)  
        kwargs.setdefault('num_surrogates', 200)          
        kwargs.setdefault('saving_path', os.getcwd())  
        kwargs.setdefault('saving', False)  
        kwargs.setdefault('saving_string', [])             

        valid_kwargs = ['Session','day','ch','dataset', 'mean_video_srate','mintimespent','minvisits','speed_threshold',
                        'nbins_pos_x','nbins_pos_y','nbins_cal','placefield_nbins_pos_x','placefield_nbins_pos_y','num_cores',
                        'num_surrogates','saving_path','saving','saving_string']
        
        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            setattr(self, k, v)
            
        self.__dict__['input_parameters'] = kwargs
        
    def main(self,mean_calcium_to_behavior,track_timevector,x_coordinates,y_coordinates):

        speed = self.get_speed(x_coordinates,y_coordinates,track_timevector)
        
        I_peaks = dp.detect_peaks(mean_calcium_to_behavior,mpd=0.5*self.mean_video_srate,mph=1.*np.nanstd(mean_calcium_to_behavior))

        x_coordinates_valid, y_coordinates_valid, mean_calcium_to_behavior_valid, track_timevector_valid = self.get_valid_timepoints(mean_calcium_to_behavior,x_coordinates,y_coordinates,track_timevector,self.speed_threshold)

        calcium_signal_binned_signal = self.get_binned_signal(mean_calcium_to_behavior_valid,self.nbins_cal)

        x_grid,y_grid,x_center_bins,y_center_bins = self.get_position_grid(x_coordinates,y_coordinates,self.nbins_pos_x,self.nbins_pos_y)

        position_binned = self.get_binned_2Dposition(x_coordinates_valid,y_coordinates_valid,x_grid,y_grid)

        nbins_pos = self.nbins_pos_x*self.nbins_pos_y
        
        entropy1 = self.get_entropy(position_binned,nbins_pos)
        
        entropy2 = self.get_entropy(calcium_signal_binned_signal,self.nbins_cal)
        
        joint_entropy = self.get_joint_entropy(position_binned,calcium_signal_binned_signal,nbins_pos,self.nbins_cal)
        
        mutualInfo_original = self.mutualInformation(entropy1,entropy2,joint_entropy)
        
        mutualInfo_permutation = self.parallelize_surrogate(position_binned,calcium_signal_binned_signal,nbins_pos,self.nbins_cal,self.num_cores,self.num_surrogates)

        mutualInfo_zscored = self.get_mutualInfo_zscore(mutualInfo_original,mutualInfo_permutation)
        
        x_grid_pc,y_grid_pc,x_center_bins_pc,y_center_bins_pc = self.get_position_grid(x_coordinates_valid,y_coordinates_valid,self.placefield_nbins_pos_x,self.placefield_nbins_pos_y)
        
        position_occupancy = self.get_occupancy(x_coordinates_valid,y_coordinates_valid,x_grid_pc,y_grid_pc,self.mean_video_srate)
        
        calcium_mean_occupancy = self.get_calcium_occupancy(mean_calcium_to_behavior_valid,x_coordinates_valid,y_coordinates_valid,x_grid_pc,y_grid_pc)
        
        visits_occupancy = self.get_visits(x_coordinates_valid,y_coordinates_valid,x_grid_pc,y_grid_pc,x_center_bins_pc,y_center_bins_pc)
        
        place_field,place_field_smoothed = self.placeField(calcium_mean_occupancy,position_occupancy,visits_occupancy,self.mintimespent, self.minvisits)
        
        sparsity = self.get_sparsity(place_field_smoothed,position_occupancy)

        matrix_output = self.get_grid_spatial_autocorrelation(place_field_smoothed)

        gridness_original = self.get_gridness_index(matrix_output)


        gridness_permutation = self.parallelize_grid_surrogate(mean_calcium_to_behavior_valid,x_coordinates_valid,
                                 y_coordinates_valid,self.placefield_nbins_pos_x,self.placefield_nbins_pos_y,
                                 self.mean_video_srate,self.mintimespent,self.minvisits,self.num_cores, self.num_surrogates) 
        
        gridness_zscored = self.get_gridness_zscored(gridness_original,gridness_permutation)
     
    
        kl_divergence_original = self.kullback_leibler_norm((place_field_smoothed).ravel(),(place_field_smoothed).ravel().shape[0])

        kl_divergence_permutation = self.parallelize_surrogate_metric2(mean_calcium_to_behavior_valid,x_coordinates_valid,y_coordinates_valid,
                                                                     position_occupancy,visits_occupancy,x_grid_pc,y_grid_pc,
                                                                     self.num_cores,self.num_surrogates)
        
                
        inputdict = dict()
        inputdict['signalMap'] = calcium_mean_occupancy
        inputdict['place_field'] = place_field
        inputdict['place_field_smoothed'] = place_field_smoothed        
        inputdict['ocuppancyMap'] = position_occupancy
        inputdict['visitsMap'] = visits_occupancy
        inputdict['x_grid'] = x_grid_pc
        inputdict['y_grid'] = y_grid_pc
        inputdict['x_center_bins'] = x_center_bins_pc
        inputdict['y_center_bins'] = y_center_bins_pc          
        inputdict['numb_events'] = I_peaks.shape[0]
        inputdict['events_index'] = I_peaks
        inputdict['mutualInfo_original'] = mutualInfo_original
        inputdict['mutualInfo_zscored'] = mutualInfo_zscored
        inputdict['mutualInfo_permutation'] = mutualInfo_permutation
        inputdict['sparsity'] = sparsity
        inputdict['gridness_permutation'] = gridness_permutation
        inputdict['gridness_original'] = gridness_original
        inputdict['gridness_zscored'] = gridness_zscored
        inputdict['kl_divergence_original'] = kl_divergence_original
        inputdict['kl_divergence_permutation'] = kl_divergence_permutation
        
        if self.saving == True:
            filename = self.Session + '.' + self.saving_string + '.PlaceField.ModulationIndex.' + self.dataset + '.Day' + str(self.day) + '.Ch.' + str(self.ch)
            self.caller_saving(inputdict,filename,self.saving_path)

            filename = self.Session + '.' + self.saving_string + '.PlaceField.Parameters.' + self.dataset + '.Day' + str(self.day) + '.Ch.' + str(self.ch)
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
     

    def get_sparsity(self,place_field,position_occupancy):
        
        position_occupancy_norm = np.nansum(position_occupancy/np.nansum(position_occupancy))
        sparsity = np.nanmean(position_occupancy_norm*place_field)**2/np.nanmean(position_occupancy_norm*place_field**2)

        
        return sparsity
    
    def get_speed(self,x_coordinates,y_coordinates,track_timevector):

        speed = np.sqrt(np.diff(x_coordinates)**2 + np.diff(y_coordinates)**2)
        speed = hf.smooth(speed/np.diff(track_timevector),window_len=10)
        speed = np.hstack([speed,0])
        return speed



    def get_position_grid(self,x_coordinates,y_coordinates,nbins_pos_x,nbins_pos_y):
        # here someone should also be able to set the enviroment edges

        x_range = (np.nanmax(x_coordinates) - np.nanmin(x_coordinates))
        x_grid_window = x_range/nbins_pos_x
        x_grid = np.arange(np.nanmin(x_coordinates),np.nanmax(x_coordinates) +x_grid_window/2,x_grid_window)

        y_range = (np.nanmax(y_coordinates) - np.nanmin(y_coordinates))
        y_grid_window = y_range/nbins_pos_y
        y_grid = np.arange(np.nanmin(y_coordinates),np.nanmax(y_coordinates)+y_grid_window/2,y_grid_window)

        x_center_bins = x_grid[0:-1] + x_grid_window/2
        y_center_bins = y_grid[0:-1] + y_grid_window/2

        return x_grid,y_grid,x_center_bins,y_center_bins


    def get_occupancy(self,x_coordinates_speed,y_coordinates_speed,x_grid,y_grid,mean_video_srate):
        # calculate position occupancy
        position_occupancy = np.zeros((y_grid.shape[0]-1,x_grid.shape[0]-1))
        for xx in range(0,x_grid.shape[0]-1):
            for yy in range(0,y_grid.shape[0]-1):

                check_x_ocuppancy = np.logical_and(x_coordinates_speed >= x_grid[xx],x_coordinates_speed < (x_grid[xx+1]))
                check_y_ocuppancy = np.logical_and(y_coordinates_speed >= y_grid[yy],y_coordinates_speed < (y_grid[yy+1]))

                position_occupancy[yy,xx] = np.sum(np.logical_and(check_x_ocuppancy,check_y_ocuppancy))/mean_video_srate

        return position_occupancy


    def get_calcium_occupancy(self,mean_calcium_to_behavior_speed,x_coordinates_speed,y_coordinates_speed,x_grid,y_grid):

        # calculate mean calcium per pixel
        calcium_mean_occupancy = np.nan*np.zeros((y_grid.shape[0]-1,x_grid.shape[0]-1)) 
        for xx in range(0,x_grid.shape[0]-1):
            for yy in range(0,y_grid.shape[0]-1):

                check_x_ocuppancy = np.logical_and(x_coordinates_speed >= x_grid[xx],x_coordinates_speed < (x_grid[xx+1]))
                check_y_ocuppancy = np.logical_and(y_coordinates_speed >= y_grid[yy],y_coordinates_speed < (y_grid[yy+1]))

                calcium_mean_occupancy[yy,xx] = np.nanmean(mean_calcium_to_behavior_speed[np.logical_and(check_x_ocuppancy,check_y_ocuppancy)])

        return calcium_mean_occupancy


    def get_visits(self,x_coordinates,y_coordinates,x_grid,y_grid,x_center_bins,y_center_bins):

            I_x_coord = []
            I_y_coord = []

            for xx in range(0,x_coordinates.shape[0]):
                I_x_coord.append(np.argmin(np.abs(x_coordinates[xx] - x_center_bins)))
                I_y_coord.append(np.argmin(np.abs(y_coordinates[xx] - y_center_bins)))

            I_x_coord = np.array(I_x_coord)
            I_y_coord = np.array(I_y_coord)

            dx = np.diff(np.hstack([I_x_coord[0]-1,I_x_coord]))
            dy = np.diff(np.hstack([I_y_coord[0]-1,I_y_coord]))

            newvisitstimes = (-1*(dy == 0))*(dx==0)+1
            newvisitstimes2 = (np.logical_or((dy != 0), (dx!=0))*1)

            I_visit = np.where(newvisitstimes>0)[0]

            # calculate visits


            x_coordinate_visit = x_coordinates[I_visit]
            y_coordinate_visit = y_coordinates[I_visit]

            visits_occupancy = np.zeros((y_grid.shape[0]-1,x_grid.shape[0]-1))        
            for xx in range(0,x_grid.shape[0]-1):
                for yy in range(0,y_grid.shape[0]-1):

                    check_x_ocuppancy = np.logical_and(x_coordinate_visit >= x_grid[xx],x_coordinate_visit < (x_grid[xx+1]))
                    check_y_ocuppancy = np.logical_and(y_coordinate_visit >= y_grid[yy],y_coordinate_visit < (y_grid[yy+1]))

                    visits_occupancy[yy,xx] = np.sum(np.logical_and(check_x_ocuppancy,check_y_ocuppancy))

            return visits_occupancy

        
    def get_valid_timepoints(self,mean_calcium_to_behavior,x_coordinates,y_coordinates,track_timevector,speed_threshold):
        
        speed = self.get_speed(x_coordinates,y_coordinates,track_timevector)

        I_speed_thres = speed > speed_threshold

        mean_calcium_to_behavior_valid = mean_calcium_to_behavior[I_speed_thres].copy()
        x_coordinates_valid = x_coordinates[I_speed_thres].copy()
        y_coordinates_valid = y_coordinates[I_speed_thres].copy()
        track_timevector_valid = track_timevector[I_speed_thres].copy()
        
        
        return x_coordinates_valid, y_coordinates_valid, mean_calcium_to_behavior_valid, track_timevector_valid
  

    def placeField(self,calcium_mean_occupancy,position_occupancy,visits_occupancy,mintimespent, minvisits):


        Valid=(position_occupancy>=mintimespent)*(visits_occupancy>=minvisits)*1.
        Valid[Valid == 0] = np.nan
        calcium_mean_occupancy = calcium_mean_occupancy*Valid

        calcium_mean_occupancy_to_smooth = np.copy(calcium_mean_occupancy)
        calcium_mean_occupancy_to_smooth[np.isnan(calcium_mean_occupancy_to_smooth)] = 0 
        calcium_mean_occupancy_smoothed = hf.gaussian_smooth_2d(calcium_mean_occupancy_to_smooth,2)

        return calcium_mean_occupancy,calcium_mean_occupancy_smoothed


    def get_binned_2Dposition(self,x_coordinates,y_coordinates,x_grid,y_grid):

        # calculate position occupancy
        position_binned = np.zeros(x_coordinates.shape) 
        count = 0
        for xx in range(0,x_grid.shape[0]-1):
            for yy in range(0,y_grid.shape[0]-1):

                check_x_ocuppancy = np.logical_and(x_coordinates >= x_grid[xx],x_coordinates < (x_grid[xx+1]))
                check_y_ocuppancy = np.logical_and(y_coordinates >= y_grid[yy],y_coordinates < (y_grid[yy+1]))
                position_binned[np.logical_and(check_x_ocuppancy,check_y_ocuppancy)] = count
                count += 1
      
        return position_binned
    

    def get_binned_signal(self,mean_calcium_to_behavior,nbins_cal):

        calcium_signal_bins = np.linspace(np.nanmin(mean_calcium_to_behavior),np.nanmax(mean_calcium_to_behavior),nbins_cal+1)
        calcium_signal_binned_signal = np.zeros(mean_calcium_to_behavior.shape[0])
        for jj in range(calcium_signal_bins.shape[0]-1):
            I_amp = (mean_calcium_to_behavior > calcium_signal_bins[jj]) & (mean_calcium_to_behavior <= calcium_signal_bins[jj+1])
            calcium_signal_binned_signal[I_amp] = jj

        return calcium_signal_binned_signal

    def mutualInformation(self,entropy1,entropy2,joint_entropy):
        eps = np.finfo(float).eps

    #     this part here could be done using this code instead. I will leave both for clarity
    #     nbins_pos = 100
    #     edges1 = np.linspace(np.nanmin(position_binned),np.nanmax(position_binned),nbins_pos+1)
    #     bin_vector1 = np.digitize(position_binned,edges1)

    #     nbins_cal = 10
    #     edges2 = np.linspace(np.nanmin(calcium_signal_binned_signal),np.nanmax(calcium_signal_binned_signal),nbins_cal+1)
    #     bin_vector2 = np.digitize(calcium_signal_binned_signal,edges2)-1

        mutualInfo = entropy1 + entropy2 - joint_entropy

        return mutualInfo

    def get_mutualInfo_zscore(self,mutualInfo_original,mutualInfo_permutation):
        mutualInfo_zscored = (mutualInfo_original-np.nanmean(mutualInfo_permutation))/np.nanstd(mutualInfo_permutation)
        return mutualInfo_zscored


    
    def get_joint_entropy(self,bin_vector1,bin_vector2,nbins_1,nbins_2):

        eps = np.finfo(float).eps

        bin_vector1 = np.copy(bin_vector1)
        bin_vector2 = np.copy(bin_vector2)

        jointprobs = np.zeros([nbins_1,nbins_2])
        
        for i1 in range(nbins_1):
            for i2 in range(nbins_2):
                jointprobs[i1,i2] = np.nansum((bin_vector1==i1) & (bin_vector2==i2))

        jointprobs = jointprobs/np.nansum(jointprobs)
        joint_entropy = -np.nansum(jointprobs*np.log2(jointprobs+eps));

        return joint_entropy
    
    
    
    def get_entropy(self,binned_input,nbins):

        eps = np.finfo(float).eps

        hdat = np.histogram(binned_input,nbins)[0]
        hdat = hdat/np.nansum(hdat)
        entropy = -np.nansum(hdat*np.log2(hdat+eps))

        return entropy

    def get_mutual_info_surrogate(self,bin_vector1,calcium_signal,nbins_1,nbins_2,permi):
        
        calcium_signal_shuffled = self.get_surrogate(calcium_signal,self.mean_video_srate)
        calcium_signal_shuffled_binned = self.get_binned_signal(calcium_signal_shuffled,self.nbins_cal)
        entropy1 = self.get_entropy(bin_vector1,nbins_1)
        entropy2 = self.get_entropy(calcium_signal_shuffled_binned,nbins_2)
        joint_entropy = self.get_joint_entropy(bin_vector1,calcium_signal_shuffled_binned,nbins_1,nbins_2)
        mutualInfo = self.mutualInformation(entropy1,entropy2,joint_entropy)
        
        
        return mutualInfo
    
    def parallelize_surrogate(self,bin_vector1,bin_vector2,nbins_1,nbins_2,num_cores,num_surrogates):
        
        results = Parallel(n_jobs=num_cores)(delayed(self.get_mutual_info_surrogate)(bin_vector1,bin_vector2,nbins_1,nbins_2,permi) for permi in range(num_surrogates))
        
        return np.array(results)
    

    def parallelize_surrogate_metric2(self,calcium_imag,x_coordinates,y_coordinates,position_occupancy,
                                      visits_occupancy,x_grid_pc,y_grid_pc,num_cores,num_surrogates):

        results = Parallel(n_jobs=num_cores)(delayed(self.get_kullback_leibler_surrogate)(calcium_imag,x_coordinates,y_coordinates,position_occupancy,visits_occupancy,x_grid_pc,y_grid_pc,permi) for permi in range(num_surrogates))

        return np.array(results)
    
    
    def kullback_leibler_norm(self,input_dist,nbin):

        input_dist_norm = (input_dist - np.nanmin(input_dist))/(np.nanmax(input_dist)-np.nanmin(input_dist))

        observed_distr = -np.nansum((input_dist_norm/np.nansum(input_dist_norm))*np.log((input_dist_norm/np.nansum(input_dist_norm))))
        test_distr = np.log(nbin);
        kl_divergence_norm = (test_distr - observed_distr) / test_distr
        return kl_divergence_norm


    
    def get_kullback_leibler_surrogate(self,calcium_imag,x_coordinates,y_coordinates,position_occupancy,
                                       visits_occupancy,x_grid_pc,y_grid_pc,permi):
        calcium_imag_shuffled = self.get_surrogate(calcium_imag,self.mean_video_srate)
        calcium_mean_occupancy_shuffled = self.get_calcium_occupancy(calcium_imag_shuffled,x_coordinates,y_coordinates,x_grid_pc,y_grid_pc)
        place_field_shuffled,place_field_smoothed_shuffled = self.placeField(calcium_mean_occupancy_shuffled,position_occupancy,visits_occupancy,self.mintimespent, self.minvisits)

        kl_divergence_surr = self.kullback_leibler_norm((place_field_smoothed_shuffled).ravel(),(place_field_smoothed_shuffled).ravel().shape[0])
        return kl_divergence_surr


    
    
    
    
#     def get_surrogate(self,input_vector,permi):
#         eps = np.finfo(float).eps
#         permi = 0
#         input_vector_shuffled = []
# #         I_break = np.random.choice(np.arange(int(input_vector.shape[0]*0.1),int(input_vector.shape[0]*0.9)),1)[0].astype(int)
#         I_break = np.random.choice(range(int(input_vector.shape[0])),1)[0]

#         if np.mod(permi,4) == 0:
#             input_vector_shuffled = np.concatenate([input_vector[I_break:], input_vector[0:I_break]])
#         elif np.mod(permi,4) == 1:
#             input_vector_shuffled = np.concatenate([input_vector[:I_break:-1], input_vector[0:I_break+1]])
#         elif np.mod(permi,4) == 2:
#             input_vector_shuffled = np.concatenate([input_vector[I_break:], input_vector[I_break-1::-1]])
#         else:   
#             input_vector_shuffled = np.concatenate([input_vector[I_break:], input_vector[0:I_break]])
#             input_vector_shuffled = input_vector_shuffled[::-1]

#         return input_vector_shuffled
    
    def get_surrogate(self,input_vector,mean_video_srate):
        eps = np.finfo(float).eps
        I_break = np.random.choice(np.arange(-10*mean_video_srate,mean_video_srate*10),1)[0].astype(int)
        input_vector_shuffled = np.concatenate([input_vector[I_break:], input_vector[0:I_break]])

        return input_vector_shuffled

    

    def parallelize_grid_surrogate(self,mean_calcium_to_behavior_valid,x_coordinates_valid,y_coordinates_valid,placefield_nbins_pos_x,placefield_nbins_pos_y,
                              mean_video_srate,mintimespent,minvisits,num_cores,num_surrogates):
        
        results = Parallel(n_jobs=num_cores)(delayed(self.get_spatial_surrogate)(mean_calcium_to_behavior_valid,x_coordinates_valid,y_coordinates_valid,placefield_nbins_pos_x,placefield_nbins_pos_y,
                              mean_video_srate,mintimespent,minvisits,permi) for permi in range(num_surrogates))
        
        return np.array(results)
    
    def get_spatial_surrogate(self,mean_calcium_to_behavior_valid,x_coordinates_valid,y_coordinates_valid,placefield_nbins_pos_x,placefield_nbins_pos_y,
                              mean_video_srate,mintimespent,minvisits,permi):
        
        mean_calcium_to_behavior_valid_shuffled = self.get_surrogate(mean_calcium_to_behavior_valid,self.mean_video_srate)
        
        x_grid_pc,y_grid_pc,x_center_bins_pc,y_center_bins_pc = self.get_position_grid(x_coordinates_valid,y_coordinates_valid,
                                                                                       placefield_nbins_pos_x,placefield_nbins_pos_y)
        
        position_occupancy = self.get_occupancy(x_coordinates_valid,y_coordinates_valid,x_grid_pc,y_grid_pc,mean_video_srate)
        
        calcium_mean_occupancy = self.get_calcium_occupancy(mean_calcium_to_behavior_valid_shuffled,x_coordinates_valid,y_coordinates_valid,x_grid_pc,y_grid_pc)
        
        visits_occupancy = self.get_visits(x_coordinates_valid,y_coordinates_valid,x_grid_pc,y_grid_pc,x_center_bins_pc,y_center_bins_pc)
        
        place_field,place_field_smoothed = self.placeField(calcium_mean_occupancy,position_occupancy,visits_occupancy,mintimespent, minvisits)
        
        matrix_output = self.get_grid_spatial_autocorrelation(place_field_smoothed)
        
        gridness = self.get_gridness_index(matrix_output)
        
        return gridness
    
    
    
    def get_gridness_index(self,array_output):

        from scipy.ndimage.interpolation import rotate


        array_output_zeroed = np.copy(array_output)
        array_output_zeroed[np.isnan(array_output_zeroed)] = 0
        autoCorr = np.copy(array_output_zeroed)
        da = 3
        angles = list(range(0, 180+da, da))
        crossCorr = []
        # Rotate and compute correlation coefficient
        for angle in angles:
            autoCorrRot = rotate(autoCorr, angle, reshape=False)
            C = np.corrcoef(np.reshape(autoCorr, (1, autoCorr.size)),
                np.reshape(autoCorrRot, (1, autoCorrRot.size)))
            crossCorr.append(C[0, 1])

        max_angles_i = (np.array([30, 90, 150]) / da).astype(int)
        min_angles_i = (np.array([60, 120]) / da).astype(int)

        maxima = np.max(np.array(crossCorr)[max_angles_i])
        minima = np.min(np.array(crossCorr)[min_angles_i])
        gridness = minima - maxima
        return gridness

    
    
    def get_grid_spatial_autocorrelation(self,input_matrix):

        matrix_image = input_matrix.copy()
        kernel = input_matrix.copy()

        [ma,na] = np.shape(matrix_image)
        [mb,nb] = np.shape(kernel)

        mc = np.max([ma+mb-1,ma,mb])
        nc = np.max([na+nb-1,na,nb])

        matrix_output = np.nan*np.zeros([mc,nc])

        i_size = kernel.shape[0]
        j_size = kernel.shape[1]

        kernel_size_i,kernel_size_j = np.shape(kernel)
        matrix_image_size = np.array(np.shape(matrix_image));

        output_matrix_size = matrix_image_size + [2*i_size-1, 2*j_size-1];
        work_mat = np.nan * np.zeros(output_matrix_size);
        work_mat[(kernel_size_i):(kernel_size_i+matrix_image_size[0]),(kernel_size_j):(kernel_size_j+matrix_image_size[1])] = matrix_image

        for i in range(np.shape(matrix_output)[0]):
            for j in range(np.shape(matrix_output)[1]):

                win1 = np.arange(i,kernel_size_i+i).astype(int)
                win2 = np.arange(j,kernel_size_j+j).astype(int)

                matrix_sliced = work_mat[win1,:][:,win2]; 
                matrix_sliced_kernel = matrix_sliced*kernel;                                             
                keep = ~np.isnan(matrix_sliced_kernel)

                n = np.sum(keep);

                if n < 20:
                    matrix_output[i,j] = np.nan;

                else:

                    sum_matrix_kernel_x_lagged = np.sum(matrix_sliced_kernel[keep]);
                    sum_matrix_lagged =   np.sum(matrix_sliced[keep]);
                    sum_matrix_kernel =   np.sum(kernel[keep]);
                    sum_matrix_lagged_2 =  np.sum(matrix_sliced[keep]**2);
                    sum_matrix_kernel_2 =  np.sum(kernel[keep]**2);

                    matrix_output[i,j] = (n*sum_matrix_kernel_x_lagged - sum_matrix_kernel*sum_matrix_lagged) / (np.sqrt(n*sum_matrix_kernel_2-sum_matrix_kernel**2) * np.sqrt(n*sum_matrix_lagged_2-sum_matrix_lagged**2));

        return matrix_output


    def get_gridness_zscored(self,gridness_original,gridness_permutation):

        gridness_zscored = (gridness_original-np.nanmean(gridness_permutation))/np.nanstd(gridness_permutation)
        return gridness_zscored