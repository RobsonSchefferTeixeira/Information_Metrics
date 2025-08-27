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
        """Return True if the array is None or all values are NaN."""
        return arr is None or np.all(np.isnan(arr))

    @staticmethod
    def initial_setup(signal_data):
        """
        Initial Setup and Conversion

        - Validates that required fields are present.
        - Converts 'input_signal', 'x_coordinates', and 'time_vector' to numpy arrays with float dtype.
        - If time_vector is empty or all NaN, it creates a linspace based on the input_signal length and sampling_rate.
        - Handles optional y_coordinates if present.
        """
        required_fields = ['input_signal', 'x_coordinates', 'time_vector']
        for field in required_fields:
            if not hasattr(signal_data, field):
                raise AttributeError(f"signal_data is missing required field: {field}")

        if DataValidator.is_empty_or_all_nan(signal_data.time_vector):
            signal_data.time_vector = np.linspace(
                0,
                len(signal_data.x_coordinates) / signal_data.sampling_rate,
                len(signal_data.x_coordinates)
            )

        # Convert to numpy arrays with float type (unless it`s spike times)
        if signal_data.signal_type == 'spikes':
            signal_data.input_signal = np.asarray(signal_data.input_signal, dtype=int)
        else:
            signal_data.input_signal = np.asarray(signal_data.input_signal, dtype=float)
        signal_data.x_coordinates = np.asarray(signal_data.x_coordinates, dtype=float)
        signal_data.time_vector = np.asarray(signal_data.time_vector, dtype=float)

        # Handle optional y_coordinates
        if hasattr(signal_data, 'y_coordinates') and signal_data.y_coordinates is not None:
            signal_data.y_coordinates = np.asarray(signal_data.y_coordinates, dtype=float)

    @staticmethod
    def validate_and_correct_shape(signal_data):
        """
        Shape Validation and Correction

        - Checks input_signal dimensions and auto-transposes if it is 2D with more cells than timesteps.
        - Determines the expected number of timesteps.
        - Validates that coordinate arrays (x_coordinates, time_vector, and optionally y_coordinates)
        are 1D and have lengths matching the number of timesteps.
        - If signal_type is 'spikes', then the signal_data must be a list in which the list len corresponds to neurons
        """
        if signal_data.signal_type == 'spikes':
            n_timesteps = len(signal_data.x_coordinates)
        # Validate input_signal shape and auto-transpose if needed
        elif signal_data.input_signal.ndim == 1:
            n_timesteps = len(signal_data.input_signal)

        elif signal_data.input_signal.ndim == 2:
            # Auto-transpose if cells > timesteps
            if signal_data.input_signal.shape[0] > signal_data.input_signal.shape[1]:
                signal_data.input_signal = signal_data.input_signal.T
            n_timesteps = signal_data.input_signal.shape[1]
        else:
            raise ValueError("input_signal must be 1D [timesteps] or 2D [cells × timesteps]")

        # Validate coordinate arrays
        has_y_coords = hasattr(signal_data, 'y_coordinates') and signal_data.y_coordinates is not None
        fields = [
            ('x_coordinates', signal_data.x_coordinates),
            ('time_vector', signal_data.time_vector)
        ]
        if has_y_coords:
            fields.append(('y_coordinates', signal_data.y_coordinates))

        for name, arr in fields:
            if arr.ndim != 1:
                raise ValueError(f"{name} must be a 1D array")
            if len(arr) != n_timesteps:
                raise ValueError(f"{name} length {len(arr)} doesn't match input_signal timesteps {n_timesteps}")
    
        if np.diff(signal_data.time_vector).min() <= 0:
            warnings.warn("time_vector is not strictly increasing",UserWarning)



    @staticmethod
    def filter_invalid_values(signal_data):
        """
        NaN and Infinite Value Filtering

        - Creates a combined validity mask from x_coordinates, time_vector, input_signal, and y_coordinates (if present).
        - If signal_type == 'spikes', remaps spike indices to the reduced valid set.
        - Raises a warning if all data points are removed.
        """
        has_y_coords = hasattr(signal_data, 'y_coordinates') and signal_data.y_coordinates is not None

        # Create combined mask from x_coordinates and time_vector
        valid_mask = np.isfinite(signal_data.x_coordinates) & np.isfinite(signal_data.time_vector)

        # Only check input_signal if not spikes
        if signal_data.signal_type != 'spikes':
            if signal_data.input_signal.ndim == 1:
                valid_mask &= np.isfinite(signal_data.input_signal)
            else:
                valid_mask &= np.all(np.isfinite(signal_data.input_signal), axis=0)

        # Update mask with y_coordinates if present
        if has_y_coords:
            valid_mask &= np.isfinite(signal_data.y_coordinates)

        # Build mapping: old_index -> new_index
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(valid_mask)[0])}

        if signal_data.signal_type == 'spikes':
            if isinstance(signal_data.input_signal, list):
                # Multi-neuron case
                filtered_signals = []
                for neuron_spikes in signal_data.input_signal:
                    valid_spikes = [old_to_new[idx] for idx in neuron_spikes if idx in old_to_new]
                    filtered_signals.append(np.array(valid_spikes, dtype=int))
                signal_data.input_signal = filtered_signals
            else:
                valid_spikes = [old_to_new[idx] for idx in signal_data.input_signal if idx in old_to_new]
                signal_data.input_signal = np.array(valid_spikes, dtype=int)
        else:
            # Continuous case
            if signal_data.input_signal.ndim == 1:
                signal_data.input_signal = signal_data.input_signal[valid_mask]
            else:
                signal_data.input_signal = signal_data.input_signal[:, valid_mask]

        # Apply filtering to aligned arrays
        signal_data.x_coordinates = signal_data.x_coordinates[valid_mask]
        signal_data.time_vector = signal_data.time_vector[valid_mask]

        if has_y_coords:
            signal_data.y_coordinates = signal_data.y_coordinates[valid_mask]

        # Warn if nothing left
        if len(signal_data.time_vector) == 0:
            warnings.warn(
                "All data points were removed during validation - check for excessive NaN/inf values",
                UserWarning
            )



    @staticmethod
    def parse_environment_edges(value):
        if value is None:
            raise ValueError("environment_edges cannot be None.")

        # Unwrap extra nesting like [[[5, 35], [40, 80]]] → [[5, 35], [40, 80]]
        while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
            value = value[0]

        # Flattened input: [5, 35] → [[5, 35]]
        if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value) and len(value) == 2:
            value = [value]

        # Must now be a list of lists
        if not (isinstance(value, list) and all(isinstance(sub, list) for sub in value)):
            raise ValueError("environment_edges must be a list of lists, e.g., [[x_min, x_max], [y_min, y_max]] or [[x_min, x_max]].")

        # Remove empty sublists
        value = [sub for sub in value if sub != []]

        if len(value) == 0:
            raise ValueError("environment_edges must contain at least x-axis bounds.")

        if len(value) > 2:
            value = value[:2]  # quietly discard extra dimensions

        for i, bounds in enumerate(value):
            if len(bounds) != 2:
                raise ValueError(f"Each sublist in environment_edges must contain exactly two numbers (index {i} is invalid).")

            min_val, max_val = bounds

            # Validate types and cast to float
            if not all(isinstance(v, (int, float)) for v in (min_val, max_val)):
                raise ValueError(f"Values in environment_edges[{i}] must be numeric. Got: {bounds}")
            min_val, max_val = float(min_val), float(max_val)

            if min_val >= max_val:
                raise ValueError(f"In environment_edges[{i}], the first value ({min_val}) must be less than the second value ({max_val}).")

            value[i] = [min_val, max_val]
    
    @staticmethod
    def validate_environment_edges(signal_data):

        if signal_data.environment_edges is None:
                
            if signal_data.y_coordinates is None:
                # 1D tracking
                x_min = np.nanmin(signal_data.x_coordinates)
                x_max = np.nanmax(signal_data.x_coordinates)    
                signal_data.environment_edges = [x_min, x_max]
            
            else:
                # 2D tracking
                x_min = np.nanmin(signal_data.x_coordinates)
                x_max = np.nanmax(signal_data.x_coordinates)
                y_min = np.nanmin(signal_data.y_coordinates)
                y_max = np.nanmax(signal_data.y_coordinates)
                signal_data.environment_edges = [[x_min, x_max], [y_min, y_max]]
        else:
                
            DataValidator.parse_environment_edges(signal_data.environment_edges)

    @staticmethod
    def get_valid_timepoints(signal_data, min_speed_threshold, min_visits, min_time_spent):

        # min speed
        I_speed_thres = signal_data.speed >= min_speed_threshold

        # min visits
        I_visits_times_thres = signal_data.visits_bins >= min_visits

        # min time spent
        I_time_spent_thres = signal_data.time_spent_inside_bins >= min_time_spent

        keep_these_frames = I_speed_thres & I_visits_times_thres & I_time_spent_thres

        # Build mapping: old_index -> new_index
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(keep_these_frames)[0])}

        if getattr(signal_data, "signal_type", None) == "spikes":
            if isinstance(signal_data.input_signal, list):
                # Multi-neuron case
                signal_data.input_signal = [
                    np.array(
                        [old_to_new[idx] for idx in neuron_spikes if idx in old_to_new],
                        dtype=int
                    )
                    for neuron_spikes in signal_data.input_signal
                ]
            else:
                # Single neuron case
                signal_data.input_signal = np.array(
                    [old_to_new[idx] for idx in signal_data.input_signal if idx in old_to_new],
                    dtype=int
                )

        else:
            # Continuous signal handling
            if signal_data.input_signal.ndim == 1:
                signal_data.input_signal = signal_data.input_signal[keep_these_frames]
            else:
                signal_data.input_signal = signal_data.input_signal[:, keep_these_frames]

        # Apply mask to all other time-aligned arrays
        signal_data.x_coordinates = signal_data.x_coordinates[keep_these_frames]

        if signal_data.y_coordinates is not None:
            signal_data.y_coordinates = signal_data.y_coordinates[keep_these_frames]

        signal_data.time_vector = signal_data.time_vector[keep_these_frames]
        signal_data.speed = signal_data.speed[keep_these_frames]
        signal_data.visits_bins = signal_data.visits_bins[keep_these_frames]
        signal_data.position_binned = signal_data.position_binned[keep_these_frames]
        signal_data.new_visits_times = signal_data.new_visits_times[keep_these_frames]
        if signal_data.bin_coordinates.ndim == 2:
            signal_data.bin_coordinates = signal_data.bin_coordinates[keep_these_frames, :]
        elif signal_data.bin_coordinates.ndim == 1:
            signal_data.bin_coordinates = signal_data.bin_coordinates[keep_these_frames]
