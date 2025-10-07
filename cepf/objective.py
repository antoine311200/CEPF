from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from scipy.optimize import minimize, root

from .distribution import Distribution
from .optimize.maxent import MaximumEntropyOptimizer

class Objective(ABC):

    @abstractmethod
    def _pdf(self, lambdas, constraints, *args, **kwargs) -> tuple[np.ndarray, float]:
        pass

    @abstractmethod
    def evaluate(self, constraints, lambdas, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def gradient(self, constraints, lambdas, *args, **kwargs) -> np.ndarray:
        pass

    def minimize(self, constraints, lambdas, tol, max_iter, method, **kwargs):
        """Minimize the objective function using L-BFGS-B"""

        result = minimize(
            fun=partial(self.evaluate, constraints, **kwargs),
            jac=partial(self.gradient, constraints, **kwargs),
            x0=lambdas,
            method=method,
            options={'gtol': tol, 'maxiter': max_iter}
        )

        # optimizer = MaximumEntropyOptimizer(
        #     A=np.array([c for c in constraints]),
        #     b=np.array([c.target for c in constraints]),
        #     max_iter=10,#max_iter,
        #     tol=tol
        # )
        # result = optimizer.optimize(lambda_init=lambdas, verbose=False)


        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)

        return {'optimal_lambdas': result.x, 'fun': result.fun, 'nfev': result.nfev}

class CrossEntropyObjective(Objective):

    def __init__(self, x: np.ndarray, dx: float) -> None:
        self.x = x
        self.dx = dx

    def _pdf(self, lambdas, constraints, prior_distribution) -> tuple[np.ndarray, float]:
        n_constraints = len(constraints)

        exponent = np.zeros_like(self.x)
        for i in range(n_constraints):
            exponent += lambdas[i] * constraints[i]

        # Avoid overflow
        exponent = np.clip(exponent, -500, 500)
        exp_exponent = np.exp(exponent) * prior_distribution.pdf
        mu = np.sum(exp_exponent) * self.dx

        if mu <= 0:
            raise ValueError("Normalization constant mu is non-positive.")

        pdf = exp_exponent / mu
        return pdf, float(mu)

    def evaluate(self, constraints, lambdas, prior_distribution: Distribution):
        """Objective function F(λ) = log(μ) - Σ λ_i d_i"""
        # Check that other_distribution has the same x and dx
        if not np.array_equal(self.x, prior_distribution.x) or self.dx != prior_distribution.dx:
            raise ValueError("Incompatible distributions")

        _, mu = self._pdf(lambdas, constraints, prior_distribution)

        targets = np.array([c.target for c in constraints])
        value = np.log(mu) - np.sum(lambdas * targets)
        return value

    def gradient(self, constraints, lambdas, prior_distribution: Distribution):
        """Gradient of the objective function ∇F(λ) = (E[c_i] - d_i)
        The assumption is that the constraints are of the form E[c_i] = d_i"""

        p, _ = self._pdf(lambdas, constraints, prior_distribution)
        n_constraints = len(constraints)

        gradient = np.zeros(n_constraints)
        for i in range(n_constraints):
            expected_value = np.sum(constraints[i] * p) * self.dx
            gradient[i] = expected_value  - constraints[i].target

        return gradient

class EntropyObjective(CrossEntropyObjective):

    def __init__(self, x: np.ndarray, dx: float) -> None:
        super().__init__(x, dx)
        self.no_prior = Distribution((x[0], x[-1]), len(x))
        self.no_prior.pdf = np.ones_like(x)

    def _pdf(self, lambdas, constraints, *args, **kwargs) -> tuple[np.ndarray, float]:
        return super()._pdf(lambdas, constraints, prior_distribution=self.no_prior)

    def evaluate(self, constraints, lambdas, *args, **kwargs):
        return super().evaluate(constraints, lambdas, prior_distribution=self.no_prior)

    def gradient(self, constraints, lambdas, *args, **kwargs):
        return super().gradient(constraints, lambdas, prior_distribution=self.no_prior)
