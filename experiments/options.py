import numpy as np
from scipy.stats import lognorm, norm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')
sns.set_context('paper')
sns.set_palette('Set2')

from cepf.minimize import MinimumCrossEntropyDistribution, MinimumEntropyDistribution
from cepf.constraint import Constraint
from cepf.distribution import Distribution

class Option:

    def __init__(self, strike: float, maturity: float, rate: float, option_type: str = 'call') -> None:
        self.strike = strike
        self.maturity = maturity
        self.rate = rate # Constant risk-free rate
        self.discount_factor = np.exp(-self.rate * maturity)

        self.option_type = option_type.lower()
        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")

    def payoff(self, S: np.ndarray) -> np.ndarray:
        if self.option_type == 'call':
            return np.maximum(S - self.strike, 0) * self.discount_factor
        else:
            return np.maximum(self.strike - S, 0) * self.discount_factor


def test_lognormal_distribution(
    S0: float,
    r: float,
    T: float,
    sigma: float,
    min_price: float,
    max_price: float,
    strikes: list[float],
    n_points: int,
    max_iter: int = 100,
    tol: float = 1e-6,
):

    # Create minimum entropy and minimum cross-entropy distribution objects
    min_entropy = MinimumEntropyDistribution(domain=(min_price, max_price), n_points=n_points)
    min_cross_entropy = MinimumCrossEntropyDistribution(domain=(min_price, max_price), n_points=n_points)

    # Compute the exact log-normal distribution for comparison
    x, dx = min_entropy.distribution.x, min_entropy.distribution.dx
    mu = np.log(S0) + (r - 0.5 * sigma**2) * T
    p_exact = lognorm.pdf(x, s=sigma * np.sqrt(T), scale=np.exp(mu))
    p_exact = p_exact / (np.sum(p_exact) * dx)

    # Compute a prior distribution
    sigma_prior = 0.35  # Higher volatility prior
    mu_prior = np.log(S0) + (r - 0.5 * sigma_prior**2) * T
    prior_pdf = lognorm.pdf(x, s=sigma_prior * np.sqrt(T), scale=np.exp(mu_prior))
    prior_pdf = prior_pdf / (np.sum(prior_pdf) * dx)

    prior_distribution = Distribution(domain=(min_price, max_price), n_points=n_points)
    prior_distribution.pdf = prior_pdf
    prior_distribution.cdf = np.cumsum(prior_pdf) * dx

    # Strikes for constraints with 0 for martingale constraint
    # The probability constraint is automatically added in the solve method
    strikes = list(set([0] + strikes))
    constraints = []
    prices = []

    for strike in strikes:
        option = Option(strike=strike, maturity=T, rate=r, option_type='call')
        payoff = option.payoff(x)

        if strike == 0:
            target = S0# * np.exp(-r * T)  # Martingale constraint
        else:
            d1 = (np.log(S0 / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            target = S0 * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)

        constraints.append(Constraint(values=payoff, target=target))
        prices.append(target)

    prices.append(1.0) # Probability constraint target

    # Solve for minimum entropy distribution
    result_entropy = min_entropy.solve(constraints, tol=tol, max_iter=max_iter)
    result_cross_entropy = min_cross_entropy.solve(constraints, tol=tol, max_iter=max_iter, prior_distribution=prior_distribution)

    return {
        "entropy": result_entropy,
        "cross_entropy": result_cross_entropy,
        "exact": p_exact,
        "prior": prior_pdf,
        "x": x,
        "dx": dx,
    }


if __name__ == "__main__":

    # Parameters for the log-normal distribution
    params = {
        "S0": 50,
        "r": 0.10,
        "T": 60 / 365,
        "sigma": 0.25,
        "min_price": 30,
        "max_price": 80,
        "n_points": 1000,
        "strikes": [40, 45, 50, 55, 60],
        "max_iter": 100,
        "tol": 1e-6,
    }

    results = test_lognormal_distribution(**params)
    entropy_dist = results["entropy"]["distribution"]
    cross_entropy_dist = results["cross_entropy"]["distribution"]

    # Get statistics
    stats_entropy = entropy_dist.get_statistics()
    stats_cross_entropy = cross_entropy_dist.get_statistics()

    # plt.figure(figsize=(12, 8))
    # Make subplot 1 taking 2/3 of the height and subplot 2 taking 1/3 of the height
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(results["x"], results["exact"], label='Exact Log-Normal', linestyle='-', color='black', linewidth=0.8, alpha=0.7)
    ax1.plot(results["x"], results["prior"], label='Prior Distribution', linestyle='--', color='gray', linewidth=0.8, alpha=0.7)
    ax1.plot(results["x"], entropy_dist.pdf, label='Min Entropy', linestyle='--', linewidth=1.5)
    ax1.plot(results["x"], cross_entropy_dist.pdf, label='Min Cross-Entropy', linestyle='-.', linewidth=1.5)

    # Highlight constraint points
    for strike in params["strikes"]:
        ax1.axvline(x=strike, color='blue', linestyle=':', linewidth=0.8, alpha=0.5)
        ax1.text(strike, ax1.get_ylim()[1]*0.1, f'Strike {strike}', rotation=90, verticalalignment='top', horizontalalignment='right', color='blue', fontsize=8)

    ax1.set_title('Minimum Entropy and Minimum Cross-Entropy Distributions of Terminal Asset Price')
    ax1.set_xlabel('Asset Price')
    ax1.set_ylabel('Probability Density')
    ax1.legend()
    ax1.grid(True)

    ax2.axis('off')
    cell_text = [
        [key, f"{value_exact:.4f}", f"{value_entropy:.4f}", f"{value_cross_entropy:.4f}"]
        for key, value_exact, value_entropy, value_cross_entropy in zip(
            stats_entropy.keys(),
            stats_entropy.values(),
            stats_cross_entropy.values(),
            stats_cross_entropy.values()
        )
    ]
    table = ax2.table(cellText=cell_text, colLabels=["Statistic", "Exact", "Entropy", "Cross-Entropy"], loc='center')
    ax2.axis('off')
    ax2.set_title("Distribution Statistics")
    plt.tight_layout()
    plt.show()
