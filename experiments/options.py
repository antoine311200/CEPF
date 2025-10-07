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


def run_experiment(
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
    method: str = 'L-BFGS-B',
    experiment: str = "lognormal",
):

    # Create minimum entropy and minimum cross-entropy distribution objects
    min_entropy = MinimumEntropyDistribution(domain=(min_price, max_price), n_points=n_points)
    min_cross_entropy = MinimumCrossEntropyDistribution(domain=(min_price, max_price), n_points=n_points)
    x, dx = min_entropy.distribution.x, min_entropy.distribution.dx

    if experiment == "lognormal":
        # Compute the exact log-normal distribution for comparison
        mu = np.log(S0) + (r - 0.5 * sigma**2) * T
        p_exact = lognorm.pdf(x, s=sigma * np.sqrt(T), scale=np.exp(mu))
        p_exact = p_exact / (np.sum(p_exact) * dx)

        exact_dist = Distribution(domain=(min_price, max_price), n_points=n_points)
        exact_dist.pdf = p_exact
        exact_dist.cdf = np.cumsum(p_exact) * dx

        # Compute a prior distribution
        sigma_prior = 0.35  # Higher volatility prior
        mu_prior = np.log(S0) + (r - 0.5 * sigma_prior**2) * T
        prior_pdf = lognorm.pdf(x, s=sigma_prior * np.sqrt(T), scale=np.exp(mu_prior))
        prior_pdf = prior_pdf / (np.sum(prior_pdf) * dx)

        prior_distribution = Distribution(domain=(min_price, max_price), n_points=n_points)
        prior_distribution.pdf = prior_pdf
        prior_distribution.cdf = np.cumsum(prior_pdf) * dx

    elif experiment == "discrete":
        init_seq = np.array([6, 5, 4, 3, 2, 1, 1, 1, 1, 1]) / 25

        # Convolve 4 times with itself to get a smoother distribution
        p_exact = np.convolve(init_seq, init_seq)
        p_exact = np.convolve(p_exact, init_seq)
        p_exact = np.convolve(p_exact, init_seq)
        p_exact = np.convolve(p_exact, init_seq)
        p_exact = p_exact / np.sum(p_exact)

        # plt.figure(figsize=(8, 4))
        # plt.stem(np.arange(len(p_exact)), p_exact, basefmt=" ")
        # plt.title("Exact Discrete Distribution")
        # plt.xlabel("Asset Price")
        # plt.ylabel("Probability")
        # plt.grid(True)
        # plt.show()

        exact_dist = Distribution(domain=(min_price, max_price), n_points=n_points)
        exact_dist.pdf = p_exact
        exact_dist.cdf = np.cumsum(p_exact) * dx

        # Prior distribution: uniform
        prior_pdf = np.ones_like(x) / (max_price - min_price)
        prior_distribution = Distribution(domain=(min_price, max_price), n_points=n_points)
        prior_distribution.pdf = prior_pdf
        prior_distribution.cdf = np.cumsum(prior_pdf) * dx
    else:
        raise ValueError("Unknown experiment")


    # Strikes for constraints with 0 for martingale constraint
    # The probability constraint is automatically added in the solve method
    strikes = list(set([0] + strikes))
    constraints = []
    prices = []

    for strike in strikes:
        option = Option(strike=strike, maturity=T, rate=r, option_type='call')
        payoff = option.payoff(x)

        if strike == 0:
            if experiment == "discrete":
                # Use the exact mean from the discrete distribution
                target = np.sum(x * exact_dist.pdf) * dx
            else:
                target = S0# * np.exp(-r * T)  # Martingale/Forward-price constraint
        else:
            if experiment == "discrete":
                # Use the exact prices from the discrete distribution
                target = np.sum(payoff * exact_dist.pdf) * dx
            else:
                # Black-Scholes price
                d1 = (np.log(S0 / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                target = S0 * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)

        constraints.append(Constraint(values=payoff, target=target))
        prices.append(target)

    prices.append(1.0) # Probability constraint target

    # Solve for minimum entropy distribution
    result_entropy = min_entropy.solve(constraints, tol=tol, max_iter=max_iter, method=method)
    result_cross_entropy = min_cross_entropy.solve(constraints, tol=tol, max_iter=max_iter, prior_distribution=prior_distribution, method=method)

    return {
        "entropy": result_entropy,
        "cross_entropy": result_cross_entropy,
        "exact": {"distribution": exact_dist},
        "prior": prior_pdf,
        "x": x,
        "dx": dx,
    }


def KL_divergence(p, q, dx):
    """Compute the Kullback-Leibler divergence D_KL(p || q)"""
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask])) * dx

def entropy(p, dx):
    """Compute the entropy H(p) = -∫ p(x) log(p(x)) dx"""
    mask = p > 0
    return -np.sum(p[mask] * np.log(p[mask])) * dx


experiment = "lognormal"  # "lognormal" or "discrete"

if __name__ == "__main__":

    # Parameters for the log-normal distribution
    params = {
        "S0": 50,
        "r": 0.10,
        "T": 60 / 365,
        "sigma": 0.25,
        "min_price": 30 if experiment == "lognormal" else 0,
        "max_price": 80 if experiment == "lognormal" else 46,
        "n_points": 1000 if experiment == "lognormal" else 46,
        "strikes": [45, 50, 55] if experiment == "lognormal" else [5, 10, 15, 20, 25],
        "max_iter": 1000,
        "tol": 1e-6,
        "method": 'L-BFGS-B' if experiment == "lognormal" else 'L-BFGS-B',
        "experiment": experiment,
    }
    results = run_experiment(**params)

    print("Lambdas for Minimum Entropy Distribution:")
    print(results["entropy"]["lambdas"])

    entropy_dist = results["entropy"]["distribution"]
    cross_entropy_dist = results["cross_entropy"]["distribution"]
    exact_dist = results["exact"]["distribution"]

    # Get statistics
    stats_entropy = entropy_dist.get_statistics()
    stats_cross_entropy = cross_entropy_dist.get_statistics()
    stats_exact = exact_dist.get_statistics()

    # Metrics
    dx = results["dx"]
    value_exact = entropy(exact_dist.pdf, dx)
    value_entropy = entropy(entropy_dist.pdf, dx)
    value_cross_entropy = entropy(cross_entropy_dist.pdf, dx)

    kl_entropy = KL_divergence(exact_dist.pdf, entropy_dist.pdf, dx)
    kl_cross_entropy = KL_divergence(exact_dist.pdf, cross_entropy_dist.pdf, dx)
    print(f"Exact Entropy: {value_exact:.6f}, Min Entropy: {value_entropy:.6f}, KL Divergence: {kl_entropy:.6f}")
    print(f"Exact Entropy: {value_exact:.6f}, Min Cross-Entropy: {value_cross_entropy:.6f}, KL Divergence: {kl_cross_entropy:.6f}")

    # RMSE
    prices_exact = []
    prices_entropy = []
    prices_cross_entropy = []
    for strike in params["strikes"]:
        option = Option(strike=strike, maturity=params["T"], rate=params["r"], option_type='call')
        payoff = option.payoff(results["x"])

        price_exact = np.sum(payoff * exact_dist.pdf) * dx
        price_entropy = np.sum(payoff * entropy_dist.pdf) * dx
        price_cross_entropy = np.sum(payoff * cross_entropy_dist.pdf) * dx

        prices_exact.append(price_exact)
        prices_entropy.append(price_entropy)
        prices_cross_entropy.append(price_cross_entropy)

    prices_exact = np.array(prices_exact)
    prices_entropy = np.array(prices_entropy)
    prices_cross_entropy = np.array(prices_cross_entropy)

    rmse_entropy = np.sqrt(np.mean((prices_exact - prices_entropy) ** 2))
    rmse_cross_entropy = np.sqrt(np.mean((prices_exact - prices_cross_entropy) ** 2))
    print(f"RMSE MED: {rmse_entropy:.6f}, RMSE MXED: {rmse_cross_entropy:.6f}")

    # Make subplot 1 taking 2/3 of the height and subplot 2 taking 1/3 of the height
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(results["x"], exact_dist.pdf, label='Exact Log-Normal', linestyle='-', color='black', linewidth=0.8, alpha=0.7)
    ax1.plot(results["x"], results["prior"], label='Prior Distribution', linestyle='--', color='gray', linewidth=0.8, alpha=0.7)
    ax1.plot(results["x"], entropy_dist.pdf, label=f'MED (rmse: {rmse_entropy:.8f})', linestyle='--', linewidth=1.5)
    ax1.plot(results["x"], cross_entropy_dist.pdf, label=f'MXED (rmse: {rmse_cross_entropy:.8f})', linestyle='-.', linewidth=1.5)

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
            stats_exact.values(),
            stats_entropy.values(),
            stats_cross_entropy.values(),
        )
    ]
    table = ax2.table(cellText=cell_text, colLabels=["Statistic", "Exact Distribution", "MED", "MXED"], loc='center')
    ax2.axis('off')
    ax2.set_title("Distribution Statistics")
    plt.tight_layout()
    plt.show()

    # Compute the smile implied volatilities for strikes from 30 to 70
    from scipy.stats import norm
    from scipy.optimize import brentq

    def black_scholes_call_price(S0, K, T, r, sigma):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def implied_volatility_call(price, S0, K, T, r):
        """Compute implied volatility using Brent's method"""
        if price < max(0, S0 - K * np.exp(-r * T)):
            return 0.0  # Arbitrage condition

        def objective(sigma):
            return black_scholes_call_price(S0, K, T, r, sigma) - price

        try:
            return brentq(objective, 1e-6, 5.0)
        except ValueError:
            return np.nan

    strike_range = np.linspace(30, 70, 20)
    iv_exact = []
    iv_entropy = []
    iv_cross_entropy = []
    for K in strike_range:
        option = Option(strike=K, maturity=params["T"], rate=params["r"], option_type='call')
        payoff = option.payoff(results["x"])

        price_exact = np.sum(payoff * exact_dist.pdf) * dx
        price_entropy = np.sum(payoff * entropy_dist.pdf) * dx
        price_cross_entropy = np.sum(payoff * cross_entropy_dist.pdf) * dx

        iv_exact.append(implied_volatility_call(price_exact, params["S0"], K, params["T"], params["r"]))
        iv_entropy.append(implied_volatility_call(price_entropy, params["S0"], K, params["T"], params["r"]))
        iv_cross_entropy.append(implied_volatility_call(price_cross_entropy, params["S0"], K, params["T"], params["r"]))

    plt.figure(figsize=(10, 5))
    plt.plot(strike_range, iv_exact, label='Exact Implied Volatility', linestyle='-', color='black', linewidth=0.8, alpha=0.7)
    plt.plot(strike_range, iv_entropy, label='MED Implied Volatility', linestyle='--', linewidth=1.5)
    plt.plot(strike_range, iv_cross_entropy, label='MXED Implied Volatility', linestyle='-.', linewidth=1.5)
    plt.title('Implied Volatility Smile')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.grid(True)
    plt.show()
