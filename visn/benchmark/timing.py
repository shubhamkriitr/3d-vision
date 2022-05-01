from functools import wraps
from visn.utils import logger
import time


def benchmark_runtime(f):
    n = 1000
    @wraps(f)
    def wrap(*args, **kw):
        total_time = 0
        for i in range(n):
            start = time.perf_counter_ns()
            result = f(*args, **kw)
            end_ = time.perf_counter_ns()
            total_time += (end_ - start)
        
        logger.info(f"Function: {f.__name__} :: Avg. "
                    f"execution time (in ns) [trials={n}] =  {total_time/n}")
        
        
        return result
    return wrap