from abc import ABC, abstractmethod


class BenchmarkObserver(ABC):
    """Base class for Benchmark Observers"""

    def register_device(self, dev):
        """Sets self.dev, for inspection by the observer at various points during benchmarking"""
        self.dev = dev

    def before_start(self):
        """before start is called every iteration before the kernel starts"""
        pass

    def after_start(self):
        """after start is called every iteration directly after the kernel was launched"""
        pass

    def during(self):
        """during is called as often as possible while the kernel is running"""
        pass

    def after_finish(self):
        """after finish is called once every iteration after the kernel has finished execution"""
        pass

    @abstractmethod
    def get_results(self):
        """get_results should return a dict with results that adds to the benchmarking data

        get_results is called only once per benchmarking of a single kernel configuration and
        generally returns averaged values over multiple iterations.
        """
        pass


class IterationObserver(BenchmarkObserver):
    pass


class ContinuousObserver(BenchmarkObserver):
    pass
