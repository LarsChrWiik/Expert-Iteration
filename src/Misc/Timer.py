
from time import time as time_now


class Timer:

    def __init__(self, time_limit):
        self.time_limit = time_limit
        self.start_time = None

    def start(self):
        self.start_time = time_now()

    def get_time_used(self):
        return time_now() - self.start_time

    def have_time_left(self):
        return self.get_time_used() < self.time_limit
