# -*- coding: utf-8 -*-
# File: prof.py


from ..utils.timer import Timer
from .base import Callback

__all__ = ['ThroughputTracker']


class ThroughputTracker(Callback):
    """
    This callback writes the training throughput (in terms of either steps/sec, or samples/sec)
    to the monitors everytime it is triggered.
    The throughput is computed based on the duration between the consecutive triggers.

    The time spent on callbacks after each epoch is excluded.
    """

    _chief_only = False

    def __init__(self, samples_per_step=None):
        """
        Args:
            samples_per_step (int or None): total number of samples processed in each step
                (i.e., your total batch size in each step).
                If not provided, this callback will record "steps/sec" instead of "samples/sec".
        """
        if samples_per_step is not None:
            samples_per_step = int(samples_per_step)
        self._samples_per_step = samples_per_step
        self._timer = Timer()
        self._timer.pause()

    # only include the time between before_epoch/after_epoch
    def _before_epoch(self):
        self._timer.resume()

    def _after_epoch(self):
        self._timer.pause()

    def _before_train(self):
        self._update_last()

    def _update_last(self):
        old_pause = self._timer.is_paused()
        self._timer.reset()
        if old_pause:
            self._timer.pause()
        self._last_step = self.global_step

    def _trigger(self):
        steps_per_sec = (self.global_step - self._last_step) / self._timer.seconds()
        self._update_last()

        if self._samples_per_step is None:
            self.trainer.monitors.put_scalar("Throughput (steps/sec)", steps_per_sec)
        else:
            self.trainer.monitors.put_scalar("Throughput (samples/sec)", steps_per_sec * self._samples_per_step)
