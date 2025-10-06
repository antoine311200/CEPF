import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from typing import Optional

from .constraint import Constraint
from .objective import Objective, CrossEntropyObjective, EntropyObjective
from .distribution import Distribution

class MinimumDistribution:

    def __init__(self, domain, n_points) -> None:
        self.distribution = Distribution(domain, n_points)
        self.entropy_value = 0.0

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
        constraints.append(Constraint(values=np.ones_like(self.distribution.x), target=1.0))
        n_constraints = len(constraints)


        if initial_lambdas is None:
            lambdas = np.zeros(n_constraints)
        else:
            lambdas = np.array(initial_lambdas)

        result = objective.minimize(constraints, lambdas, tol, max_iter, *args, **kwargs)
        lambdas = result["optimal_lambdas"]

        self.distribution.pdf, _ = objective._pdf(constraints, lambdas, **kwargs)
        self.distribution.cdf = np.cumsum(self.distribution.pdf) * self.distribution.dx

        return {
            "lambdas": lambdas,
            "distribution": self.distribution,
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
        prior_distribution: Optional[Distribution] = None,
        tol=1e-6,
        max_iter=100,
    ):
        objective = CrossEntropyObjective(self.distribution.x, self.distribution.dx)
        return super().solve(constraints, objective, initial_lambdas, tol, max_iter, prior_distribution=prior_distribution)
