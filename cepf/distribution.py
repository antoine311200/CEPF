import numpy as np

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

        entropy = -self._expected_value(lambda x: np.log(self.pdf + 1e-15))

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
            "kurtosis": kurtosis,
            "entropy": entropy
        }

class PiecewiseTiltedDistribution(Distribution):
    """Piece-wise tilted distribution defined by knots and values at knots"""

    def __init__(self, domain, n_points, knots, values) -> None:
        super().__init__(domain, n_points)

        if len(knots) != len(values):
            raise ValueError("Knots and values must have the same length")
        if np.any(np.diff(knots) <= 0):
            raise ValueError("Knots must be in strictly increasing order")
        # if knots[0] > domain[0] or knots[-1] < domain[1]:
        #     raise ValueError("Knots must cover the entire domain")

        self.knots = knots
        self.values = values

        self.pdf = self._compute_pdf()
        self.cdf = np.cumsum(self.pdf) * self.dx

    def _compute_pdf(self):
        """Compute the PDF as value[0] exp(- value[1] * x) piece-wise"""
        pdf = np.zeros_like(self.x)

        for i in range(len(self.knots) - 1):
            mask = (self.x >= self.knots[i]) & (self.x < self.knots[i + 1])
            pdf[mask] = self.values[i][0] * np.exp(-self.values[i][1] * self.x[mask])

        # Handle the last knot to include the endpoint
        if len(self.knots) > 1:
            mask = (self.x >= self.knots[-2]) & (self.x <= self.knots[-1])
            pdf[mask] = (1 / self.values[-1][1]) * np.exp(-self.values[-1][1] * self.x[mask])
        else:
            pdf = self.values[0][0] * np.exp(-self.values[0][1] * self.x)

        # Normalize the PDF
        pdf_sum = np.sum(pdf) * self.dx
        if pdf_sum > 0:
            pdf /= pdf_sum
        return pdf

class TiltedDistribution(PiecewiseTiltedDistribution):
    """Exponentially tilted distribution defined by parameters (alpha, beta)"""

    def __init__(self, domain, n_points, alpha, beta) -> None:
        super().__init__(domain, n_points, knots=[domain[0]], values=[(alpha, beta)])

        if beta <= 0:
            raise ValueError("Beta must be positive")

        self.alpha = alpha
        self.beta = beta

        self.pdf = self._compute_pdf()
        self.cdf = np.cumsum(self.pdf) * self.dx