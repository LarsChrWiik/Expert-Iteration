
from time import time as time_now


class Timer:

    def __init__(self):
        self.start_time = None
        self.search_time = None

    def start(self, search_time):
        self.start_time = time_now()
        self.search_time = search_time

    def have_time_left(self):
        return time_now() - self.start_time < self.search_time
