import numpy as np
import matplotlib.pyplot as plt

from cepf.distribution import PiecewiseTiltedDistribution, TiltedDistribution, Distribution

def show_distribtion(*distributions, **kwargs):
    for i, distribution in enumerate(distributions):
        print(f"Statistics for Distribution {i+1}:")
        stats = distribution.get_statistics()

        for key, value in stats.items():
            print(f"{key}: {value}", end=" | ")
        print()

    labels = kwargs.get('labels', [f'Distribution {i+1}' for i in range(len(distributions))])

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for i, distribution in enumerate(distributions):
        plt.plot(distribution.x, distribution.pdf, label=labels[i])
    plt.title('Piecewise Tilted Distribution PDF')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, distribution in enumerate(distributions):
        plt.plot(distribution.x, distribution.cdf, label=labels[i])
    plt.title('Piecewise Tilted Distribution CDF')
    plt.xlabel('x')
    plt.ylabel('Cumulative Probability')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_points = 1000

    # knots = [0, 200, 400, 600, 800, 1000]
    # values = [(0.1, 0.001), (0.8, 0.002), (0.6, 0.003), (0.4, 0.004), (0.2, 0.005), (0.1, 0.006)]
    # domain = (knots[0], knots[-1] + knots[1] - knots[0])  # Extend domain to the right

    # pwd = PiecewiseTiltedDistribution(domain, n_points, knots, values)
    # show_distribtion(pwd)

    # etd = PiecewiseTiltedDistribution((0, 1000), n_points, [0], [(0.1, 0.001)])
    # show_distribtion(etd)

    # td = TiltedDistribution((0, 1000), n_points, 0.1, 0.001)
    # show_distribtion(td)

    prior = Distribution((0, 1000), n_points)
    init_seq = np.array([6, 5, 4, 3, 2, 1, 1, 1, 1, 1]) / 25
    prior_pdf = np.convolve(np.convolve(np.convolve(np.convolve(init_seq, init_seq), init_seq), init_seq), init_seq)
    prior_pdf = prior_pdf / np.sum(prior_pdf)
    prior.pdf = np.interp(prior.x, np.linspace(0, 1000, len(prior_pdf)), prior_pdf)
    prior.pdf /= np.sum(prior.pdf) * prior.dx
    prior.cdf = np.cumsum(prior.pdf) * prior.dx

    knots = [0, 200, 400, 600, 800, 1000]
    values = [(0.1, 0.001), (0.8, 0.002), (0.6, 0.003), (0.4, 0.004), (0.2, 0.005), (0.1, 0.006)]
    domain = (knots[0], knots[-1])

    pwd_with_prior = PiecewiseTiltedDistribution(domain, n_points, knots, values, prior_distribution=prior)
    pwd = PiecewiseTiltedDistribution(domain, n_points, knots, values)
    show_distribtion(prior, pwd, pwd_with_prior, labels=['Prior', 'Piecewise Tilted', 'Piecewise Tilted with Prior'])