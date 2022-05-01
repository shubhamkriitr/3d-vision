from visn.utils import logger
import time

class BenchMarker(object):
    def __init__(self, config=None) -> None:
        self.config = config
        self.default_num_trials = 1000
    
    def compare_runtimes(self, methods, args_kwargs,
                               use_same_args_kwargs_for_all, num_trials=None,
                               normalization_mode: str=None):
        """
        `methods`: a list of methods to find execution time of
        `args_kwargs`: if `use_same_args_kwargs_for_all` is `True`, then 
            it should be a tuple of (args, kwargs) that would passed to each of
            the method `f`(say) in `methods` like `f(*args, **kwargs)`.
            If `use_same_args_kwargs_for_all` is `False`, then it should
            be a list of tuples (mentioned earlier), one tuple for each method.
        `normalization_mode`: Indicate how to normalize runtime values
             'first_element' => scale values such that first element
            of the returned list is 1.
            'min' => scale such that miminum value is set to 1
            
        """
        
        exec_times_ns = [] # runtimes in ns
        args, kwargs = None, None
        if use_same_args_kwargs_for_all:
            args, kwargs = args_kwargs
        for idx, f in enumerate(methods):
            if not use_same_args_kwargs_for_all:
                args, kwargs = args_kwargs[idx]
            time_taken = self.get_runtime_ns(f, args, kwargs, num_trials=num_trials)
            exec_times_ns.append(time_taken)
        
        if normalization_mode is not None:
            # normalize
            return self.normalize_values(exec_times_ns, normalization_mode)
        
        return exec_times_ns
    
    def normalize_values(self, values: list, mode:str = "first_element"):
        if len(values) == 0:
            return values
        if mode == "first_element":
            scale_down_by = values[0]
        elif mode == "min":
            scale_down_by = min(values)
        else:
            raise ValueError(f"Invalid mode : {mode}")
        
        return [v/scale_down_by for v in values]
        
        
    
    def get_runtime_ns(self, f, args: list, kwargs: dict,
                       num_trials=None):
        if num_trials is None:
            num_trials = self.default_num_trials
        total_time = 0
        for i in range(num_trials):
            start = time.perf_counter_ns()
            f(*args, **kwargs)
            end_ = time.perf_counter_ns()
            total_time += (end_ - start)
        
        average_time = total_time/num_trials
        logger.debug(f"Function: {f.__name__} :: Avg. "
                    f"execution time (in ns) [trials={num_trials}] "
                    f"=  {average_time}")
        
        return average_time
