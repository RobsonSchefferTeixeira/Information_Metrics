import inspect

class ParameterValidator:

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
            if not isinstance(value, (int)):
                raise TypeError("x_bin_size must be of type int.")
            if value <= 0:
                raise ValueError("x_bin_size must be a positive number.")

    @staticmethod
    def validate_y_bin_size(value):
        if value is not None:
            if not isinstance(value, (int)):
                raise TypeError("y_bin_size must be of type int.")
            if value <= 0:
                raise ValueError("y_bin_size must be a positive number.")
            


    @staticmethod
    def validate_smoothing_size(value):
        if value is not None:
            if not isinstance(value, (int)):
                raise TypeError("smoothing_size must be of type int.")
            if value <= 0:
                raise ValueError("smoothing_size must be a positive number.")
            

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
    def validate_speed_smoothing_points(value):
        if value is not None:
            if not isinstance(value, (int)):
                raise TypeError("speed_smoothing_points must be of type int.")                           
            if value <= 0:
                raise ValueError("speed_smoothing_points must be a positive number.")

    @staticmethod
    def validate_detection_threshold(value):
        if value is not None:
            if not isinstance(value, (int,float)):
                raise TypeError("detection_threshold must be of type int of float.")
            if value <= 0:
                raise ValueError("detection_threshold must be a positive number.")
            
    @staticmethod
    def validate_detection_smoothing_size(value):
        if value is not None:
            if not isinstance(value, (int)):
                raise TypeError("detection_smoothing_size must be of type int.")
            if value <= 0:
                raise ValueError("detection_smoothing_size must be a positive number.")

    @staticmethod
    def validate_nbins_cal(value):
        if value is not None:
            if not isinstance(value, (int)):
                raise TypeError("nbins_cal must be of type int.")
            if value <= 1:
                raise ValueError("nbins_cal must be a positive number higher than 1.")




    @staticmethod
    def validate_environment_edges(value):
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
