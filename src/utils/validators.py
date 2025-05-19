import inspect
import warnings
import numpy as np

class ParameterValidator:

    @staticmethod
    def validate_num_cores(value):
        if value is not None:
            if not isinstance(value, (int,float)):
                raise TypeError("num_cores must be of type int or float.")
            if value == 0:
                raise ValueError("num_cores cannot be zero.")


    @staticmethod
    def validate_sampling_rate(value):
        if value is not None:
            if not isinstance(value, (int, float)):
                raise TypeError("sampling_rate must be of type int or float.")
            if value <= 0:
                raise ValueError("sampling_rate must be a positive number.")

    @staticmethod
    def validate_num_surrogates(value):
        if value is not None:
            if not isinstance(value, (int)):
                raise TypeError("num_surrogates must be of type int.")
            if value <= 0:
                raise ValueError("num_surrogates must be a positive number.")
            
    @staticmethod
    def validate_min_time_spent(value):
        if value is not None:
            if not isinstance(value, (int, float)):
                raise TypeError("min_time_spent must be of type int or float.")
            if value < 0:
                raise ValueError("min_time_spent must be a non-negative number.")

    @staticmethod
    def validate_field_detection_method(value):
        if value is not None:
            if not isinstance(value, str):
                raise TypeError("field_detection_method must be of type str: std_from_field or random_fields.")
            if value not in ['std_from_field', 'random_fields']:
                warnings.warn("No valid field detection method set. Use std_from_field or random_fields", UserWarning)



    @staticmethod
    def validate_min_visits(value):
        if value is not None:
            if not isinstance(value, (int)):
                raise TypeError("min_visits must be of type int.")
            if value < 0:
                raise ValueError("min_visits must be a non-negative number.")
            

    @staticmethod
    def validate_min_speed_threshold(value):
        if value is not None:
            if not isinstance(value, (int,float)):
                raise TypeError("min_speed_threshold must be of type int or float.")
            if value < 0:
                raise ValueError("min_speed_threshold must be a non-negative number.")
    
    @staticmethod
    def validate_x_bin_size(value):
        if value is not None:
            if not isinstance(value, (int,float)):
                raise TypeError("x_bin_size must be of type int or float.")
            if value <= 0:
                raise ValueError("x_bin_size must be a positive number.")

    @staticmethod
    def validate_y_bin_size(value):
        if value is not None:
            if not isinstance(value, (int,float)):
                raise TypeError("y_bin_size must be of type int or float.")
            if value <= 0:
                raise ValueError("y_bin_size must be a positive number.")
            


    @staticmethod
    def validate_map_smoothing_sigma(value):
        if value is not None:
            if not isinstance(value, (int,float)):
                raise TypeError("map_smoothing_sigma must be of type int or float.")
            if value <= 0:
                raise ValueError("map_smoothing_sigma must be a positive number.")
            

    @staticmethod
    def validate_shift_time(value):
        if value is not None:
            if not isinstance(value, (int,float)):
                raise TypeError("shift_time must be of type int of float.")
            if value <= 0:
                raise ValueError("shift_time must be a positive number.")

 
    @staticmethod
    def validate_saving(value):
        if value is not None:
            if not isinstance(value, (bool)):
                raise ValueError("saving must be boolean (True of False).")
            
            
    @staticmethod
    def validate_percentile_threshold(value):
        if value is not None:
            if not isinstance(value, (int,float)):
                raise TypeError("percentile_threshold must be of type int of float.")
            if (value < 0) or (value > 100):
                raise ValueError("percentile_threshold must be a positive number between 0 and 100.")
    
    @staticmethod
    def validate_min_num_of_bins(value):
        if value is not None:
            if not isinstance(value, (int)):
                raise TypeError("min_num_of_bins must be of type int.")            
            if value <= 0:
                raise ValueError("min_num_of_bins must be a positive number.")


    @staticmethod
    def validate_speed_smoothing_sigma(value):
        if value is not None:
            if not isinstance(value, (int,float)):
                raise TypeError("speed_smoothing_sigma must be of type int or float.")                           
            if value <= 0:
                raise ValueError("speed_smoothing_sigma must be a positive number.")

    @staticmethod
    def validate_detection_threshold(value):
        if value is not None:
            if not isinstance(value, (int,float)):
                raise TypeError("detection_threshold must be of type int of float.")
            if value <= 0:
                raise ValueError("detection_threshold must be a positive number.")
            
    @staticmethod
    def validate_detection_smoothing_sigma(value):
        if value is not None:
            if not isinstance(value, (int,float)):
                raise TypeError("detection_smoothing_sigma must be of type int or float.")
            if value <= 0:
                raise ValueError("detection_smoothing_sigma must be a positive number.")

    @staticmethod
    def validate_nbins_cal(value):
        if value is not None:
            if not isinstance(value, (int)):
                raise TypeError("nbins_cal must be of type int.")
            if value <= 1:
                raise ValueError("nbins_cal must be a positive number higher than 1.")


    @classmethod
    def validate_all(cls, params):
        for key, validator in cls.get_validators().items():
            validator(params.get(key))

    @classmethod
    def get_validators(cls):
        validators = {}
        for name, func in inspect.getmembers(cls, inspect.isfunction):
            # Check if the function is a static method by inspecting the class __dict__
            if isinstance(cls.__dict__.get(name), staticmethod) and name.startswith('validate_'):
                param_name = name[len('validate_'):]  # Extract parameter name
                validators[param_name] = func
        return validators




class DataValidator:

    @staticmethod
    def is_empty_or_all_nan(arr):
        return arr.size == 0 or np.isnan(arr).all()

    @staticmethod
    def validate_input_data(signal_data):
        """
        Comprehensive validation and cleaning of input data that:
        1. Validates array shapes and dimensions
        2. Handles optional y_coordinates
        3. Ensures float dtype
        4. Automatically transposes 2D input_signal if needed
        5. Removes NaN and infinite values across all arrays
        
        Parameters:
            signal_data (object): Input data object containing:
                - input_signal (np.ndarray): 1D [timesteps] or 2D [cells × timesteps]
                - x_coordinates (np.ndarray): 1D array of x positions
                - y_coordinates (np.ndarray, optional): 1D array of y positions
                - time_vector (np.ndarray): 1D array of timestamps
        """
        # ======================
        # 1. Initial Setup and Validation
        # ======================
        
        # Check required fields
        required_fields = ['input_signal', 'x_coordinates', 'time_vector']
        for field in required_fields:
            if not hasattr(signal_data, field):
                raise AttributeError(f"signal_data is missing required field: {field}")
        
        if DataValidator.is_empty_or_all_nan(signal_data.time_vector):
            signal_data.time_vector = np.linspace(0,len(signal_data.input_signal)/signal_data.sampling_rate,len(signal_data.input_signal))

        # Convert to numpy arrays and ensure float type
        signal_data.input_signal = np.asarray(signal_data.input_signal, dtype=float)
        signal_data.x_coordinates = np.asarray(signal_data.x_coordinates, dtype=float)
        signal_data.time_vector = np.asarray(signal_data.time_vector, dtype=float)
        
        # Handle optional y_coordinates
        has_y_coords = hasattr(signal_data, 'y_coordinates') and signal_data.y_coordinates is not None
        if has_y_coords:
            signal_data.y_coordinates = np.asarray(signal_data.y_coordinates, dtype=float)
        
        # ======================
        # 2. Shape Validation and Correction
        # ======================
        
        # Validate input_signal shape and auto-transpose if needed
        if signal_data.input_signal.ndim == 1:
            n_timesteps = len(signal_data.input_signal)
        elif signal_data.input_signal.ndim == 2:
            # Auto-transpose if cells > timesteps
            if signal_data.input_signal.shape[0] > signal_data.input_signal.shape[1]:
                signal_data.input_signal = signal_data.input_signal.T
            n_timesteps = signal_data.input_signal.shape[1]
        else:
            raise ValueError("input_signal must be 1D [timesteps] or 2D [cells × timesteps]")
        
        # Validate coordinate/time dimensions
        for name, arr in [('x_coordinates', signal_data.x_coordinates),
                        ('time_vector', signal_data.time_vector)] + \
                        ([('y_coordinates', signal_data.y_coordinates)] if has_y_coords else []):
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1D array")
            if len(arr) != n_timesteps:
                raise ValueError(f"{name} length {len(arr)} doesn't match input_signal timesteps {n_timesteps}")
        
        # ======================
        # 3. NaN and Infinite Value Filtering
        # ======================
        

        # Create combined validity mask
        valid_mask = np.isfinite(signal_data.x_coordinates) & np.isfinite(signal_data.time_vector)
        
        # Handle input_signal based on dimensionality
        if signal_data.input_signal.ndim == 1:
            valid_mask &= np.isfinite(signal_data.input_signal)
        else:
            valid_mask &= np.all(np.isfinite(signal_data.input_signal), axis=0)
        
        # Include y_coordinates in mask if present
        if has_y_coords:
            valid_mask &= np.isfinite(signal_data.y_coordinates)
        
        # I need to improve this. Let`s deactivate it for now`
        # Apply filtering
        if signal_data.input_signal.ndim == 1:
            signal_data.input_signal = signal_data.input_signal[valid_mask]
        else:
            signal_data.input_signal = signal_data.input_signal[:, valid_mask]
        
        signal_data.x_coordinates = signal_data.x_coordinates[valid_mask]
        signal_data.time_vector = signal_data.time_vector[valid_mask]
        
        if has_y_coords:
            signal_data.y_coordinates = signal_data.y_coordinates[valid_mask]
        
        # Final validation check
        if len(signal_data.time_vector) == 0:
            warnings.warn("All data points were removed during validation - check for excessive NaN/inf values")



    @staticmethod
    def validate_environment_edges_format(value):
        if value is not None:
            if isinstance(value, list):
                if all(isinstance(sublist, list) for sublist in value):
                    # Handle case where value is a list of sublists
                    if len(value) != 2:
                        raise ValueError("environment_edges must contain exactly two sublists.")
                    for sublist in value:
                        if len(sublist) != 2:
                            raise ValueError("Each sublist in environment_edges must contain exactly two numbers.")
                        min_val, max_val = sublist
                        if not all(isinstance(i, (int, float)) for i in (min_val, max_val)):
                            raise ValueError("Each value in the sublists must be a number.")
                        if min_val >= max_val:
                            raise ValueError("In each sublist, the first value (min) must be less than the second value (max).")
                else:
                    # Handle case where value is a single list
                    if len(value) != 2:
                        raise ValueError("environment_edges must contain exactly two numbers.")
                    min_val, max_val = value
                    if not all(isinstance(i, (int, float)) for i in (min_val, max_val)):
                        raise ValueError("Each value in environment_edges must be a number.")
                    if min_val >= max_val:
                        raise ValueError("In environment_edges, the first value (min) must be less than the second value (max).")
            else:
                raise ValueError("environment_edges must be a list.")


    def validate_environment_edges(signal_data,environment_edges):

        DataValidator.validate_environment_edges_format(environment_edges)

        if environment_edges is None:
                
            if signal_data.y_coordinates is None:
                # 1D tracking
        
                x_min = np.nanmin(signal_data.x_coordinates)
                x_max = np.nanmax(signal_data.x_coordinates)    
                signal_data.environment_edges = [[x_min, x_max]]
            
            else:
                # 2D tracking
                x_min = np.nanmin(signal_data.x_coordinates)
                x_max = np.nanmax(signal_data.x_coordinates)
                y_min = np.nanmin(signal_data.y_coordinates)
                y_max = np.nanmax(signal_data.y_coordinates)
                signal_data.environment_edges = [[x_min, x_max], [y_min, y_max]]
        else:
            signal_data.environment_edges = environment_edges

    

    @staticmethod
    def get_valid_timepoints(signal_data, min_speed_threshold, min_visits, min_time_spent):
   
        # min speed
        I_speed_thres = signal_data.speed >= min_speed_threshold

        # min visits
        I_visits_times_thres = signal_data.visits_bins >= min_visits

        # min time spent
        I_time_spent_thres = signal_data.time_spent_inside_bins >= min_time_spent

        keep_these_frames = I_speed_thres * I_visits_times_thres * I_time_spent_thres
            
        # Handle input_signal (works for both 1D and 2D)
        if signal_data.input_signal.ndim == 1:
            signal_data.input_signal = signal_data.input_signal[keep_these_frames]
        else:
            signal_data.input_signal = signal_data.input_signal[:, keep_these_frames]
                    
        signal_data.x_coordinates = signal_data.x_coordinates[keep_these_frames]

        if signal_data.y_coordinates is not None:
            signal_data.y_coordinates = signal_data.y_coordinates[keep_these_frames]

        signal_data.time_vector = signal_data.time_vector[keep_these_frames]
        signal_data.speed = signal_data.speed[keep_these_frames]
        signal_data.visits_bins = signal_data.visits_bins[keep_these_frames]
        signal_data.position_binned = signal_data.position_binned[keep_these_frames]
        signal_data.new_visits_times = signal_data.new_visits_times[keep_these_frames]

    
