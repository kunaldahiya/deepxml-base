import time
import functools


def timeit(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      
        run_time = end_time - start_time
        print("Finished {} in {:.4f} secs".format(func.__name__, run_time))
        return value
    return wrapper_timer


class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0
        self.running = False

    def tic(self):
        """Start or resume the timer."""
        if not self.running:
            self.start_time = time.time()
            self.running = True

    def toc(self):
        """Pause the timer and return the elapsed time."""
        if self.running:
            self.elapsed_time += time.time() - self.start_time
            self.running = False
        return self.elapsed_time

    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.elapsed_time = 0
        self.running = False