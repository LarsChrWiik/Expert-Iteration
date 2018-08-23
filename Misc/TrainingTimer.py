
from time import time as time_now


def get_seconds(d=0, h=0, m=0, s=0, ms=0):
    """
    Converts inputs to seconds.

    :param d: Days.
    :param h: Hours.
    :param m: Minutes.
    :param s: Seconds.
    :param ms: Milliseconds.

    :return: float representing seconds.
    """
    if (d, h, m, s) == (0, 0, 0):
        raise Exception("Cannot return 0 seconds. ")
    return s + 60*m + 60*60*h + 60*60*24*d + 0.001*ms


class TrainingTimer:

    def __init__(self, time_limit, num_versions=1):
        self.time_limit = time_limit
        self.num_versions = num_versions
        self.version_time = time_limit / num_versions
        self.start_time = None
        self.last = None

    def start_new_lap(self):
        self.start_time = time_now()
        self.last = time_now()

    def get_time_used(self):
        return time_now() - self.start_time

    def get_time_since_last_check(self):
        tn = time_now()
        time_used = tn - self.last
        self.last = tn
        return time_used

    def has_time_left(self):
        return self.get_time_used() < self.version_time
