import time

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