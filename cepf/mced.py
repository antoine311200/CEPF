import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from .constraint import Constraint
from .objective import Objective, CrossEntropyObjective, EntropyObjective
from .distribution import Distribution

class MinimumDistribution:

    def __init__(self, domain, n_points) -> None:
        self.distribution = Distribution(domain, n_points)
        self.entropy_value = 0.0

    def entropy(self):
        """Calculate the entropy of the distribution"""
        if self.entropy_value == 0.0:
            safe_pdf = np.clip(self.pdf, 1e-12, None)
            self.entropy_value = -np.sum(safe_pdf * np.log(safe_pdf + 1e-12))
        return self.entropy_value

    def cross_entropy(self, other):
        """Calculate the cross-entropy with another distribution"""
        safe_pdf = np.clip(self.pdf, 1e-12, None)
        safe_other_pdf = np.clip(other.pdf, 1e-12, None)
        return -np.sum(safe_pdf * np.log(safe_other_pdf + 1e-12))

    def solve(
        self,
        constraints: list[Constraint],
        objective: Objective,
        initial_lambdas=None,
        tol=1e-6,
        max_iter=100,
        *args,
        **kwargs
    ):
        n_constraints = len(constraints)

        if initial_lambdas is None:
            lambdas = np.zeros(n_constraints)
        else:
            lambdas = np.array(initial_lambdas)

        result = objective.minimize(constraints, lambdas, tol, max_iter)
        lambdas = result["optimal_lambdas"]

        n_constraints = len(constraints)
        exponent = np.zeros_like(self.distribution.x)
        for i in range(n_constraints):
            exponent += lambdas[i] * constraints[i]

        # Avoid overflow
        exponent = np.clip(exponent, -500, 500)
        exp_exponent = np.exp(exponent)
        mu = np.sum(exp_exponent) * self.distribution.dx

        self.pdf = exp_exponent / mu
        self.cdf = np.cumsum(self.pdf) * self.distribution.dx

        return {
            "lambdas": lambdas,
            "pdf": self.pdf,
            "cdf": self.cdf,
            "objective_value": result["fun"],
            "nfev": result["nfev"],
        }

class MinimumEntropyDistribution(MinimumDistribution):

    def solve(
        self,
        constraints: list[Constraint],
        initial_lambdas=None,
        tol=1e-6,
        max_iter=100,
    ):
        objective = EntropyObjective(self.distribution.x, self.distribution.dx)
        return super().solve(constraints, objective, initial_lambdas, tol, max_iter)


class MinimumCrossEntropyDistribution(MinimumDistribution):

    def solve(
        self,
        constraints: list[Constraint],
        initial_lambdas=None,
        tol=1e-6,
        max_iter=100,
    ):
        objective = CrossEntropyObjective(self.distribution.x, self.distribution.dx)
        return super().solve(constraints, objective, initial_lambdas, tol, max_iter)
