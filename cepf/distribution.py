import numpy as np
from scipy.interpolate import interp1d

class Distribution:

    def __init__(self, domain, n_points) -> None:
        self.domain = domain
        self.n_points = n_points

        self.x = np.linspace(domain[0], domain[1], n_points)
        self.dx = self.x[1] - self.x[0]

        self.pdf = np.zeros(n_points)
        self.cdf = np.zeros(n_points)

    def _expected_value(self, func):
        """Calculate the expected value of a function under the distribution"""
        return np.sum(func(self.x) * self.pdf) * self.dx

    def get_mean(self):
        """Calculate the mean of the distribution"""
        return self._expected_value(lambda x: x)

    def get_variance(self):
        """Calculate the variance of the distribution"""
        mean = self.get_mean()
        return self._expected_value(lambda x: (x - mean)**2)

    def get_std(self):
        """Calculate the standard deviation of the distribution"""
        return np.sqrt(self.get_variance())

    def get_skewness(self):
        """Calculate the skewness of the distribution"""
        return self._expected_value(lambda x: ((x - self.get_mean()) / self.get_std())**3)

    def get_kurtosis(self):
        """Calculate the kurtosis of the distribution"""
        return self._expected_value(lambda x: ((x - self.get_mean()) / self.get_std())**4)

    def get_statistics(self):
        """Calculate mean and variance of the distribution"""
        mean = self.get_mean()
        variance = self.get_variance()
        std = np.sqrt(variance)

        if std > 0:
            skewness = self._expected_value(lambda x: ((x - mean)/std)**3)
            kurtosis = self._expected_value(lambda x: ((x - mean)/std)**4)
        else:
            skewness = 0
            kurtosis = 0

        return {
            "mean": mean,
            "variance": variance,
            "std": std,
            "skewness": skewness,
            "kurtosis": kurtosis
        }
