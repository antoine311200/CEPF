import matplotlib.pyplot as plt

from cepf.distribution import PiecewiseTiltedDistribution, TiltedDistribution

def show_distribtion(distribution):
    stats = distribution.get_statistics()

    print("Piecewise Tilted Distribution Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(distribution.x, distribution.pdf, label='PDF')
    plt.title('Piecewise Tilted Distribution PDF')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(distribution.x, distribution.cdf, label='CDF', color='orange')
    plt.title('Piecewise Tilted Distribution CDF')
    plt.xlabel('x')
    plt.ylabel('Cumulative Probability')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_points = 1000
    knots = [0, 200, 400, 600, 800, 1000]
    values = [(0.1, 0.001), (0.8, 0.002), (0.6, 0.003), (0.4, 0.004), (0.2, 0.005), (0.1, 0.006)]
    domain = (knots[0], knots[-1] + knots[1] - knots[0])  # Extend domain to the right

    pwd = PiecewiseTiltedDistribution(domain, n_points, knots, values)
    show_distribtion(pwd)

    etd = PiecewiseTiltedDistribution((0, 1000), n_points, [0], [(0.1, 0.001)])
    show_distribtion(etd)

    td = TiltedDistribution((0, 1000), n_points, 0.1, 0.001)
    show_distribtion(td)