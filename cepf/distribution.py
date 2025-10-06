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

    # # Not sure if this is the best way to do it
    # def pdf(self):
    #     """Probability density function"""
    #     interpolator = interp1d(self.x, self._pdf, kind='linear', fill_value=0.0, bounds_error=False)
    #     pdf_func = lambda x: interpolator(x)
    #     return pdf_func

    # def cdf(self):
    #     """Cumulative distribution function"""
    #     interpolator = interp1d(self.x, self._cdf, kind='linear', fill_value=0.0, bounds_error=False)
    #     cdf_func = lambda x: interpolator(x)
    #     return cdf_func