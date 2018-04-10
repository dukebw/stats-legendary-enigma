"""Find the MLE of a ZIP (Zero-Inflated Poisson) distribution using Expectation
Maximization.
"""
import csv
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats


class Struct:
    """A convenient struct-like class.

    Source: Python Cookbook, Beazley and Jones. 3rd Edition. 8.11. Simplifying
        the Initialization of Data Structures.

    Usage:
        ```python
        class Stock(Structure):
            _fields = ['name', 'shares', 'price']

        s1 = Stock('ACME', 50, 91.1)
        s2 = Stock('ACME', 50, price=91.1)
        s3 = Stock('ACME', shares=50, price=91.1)
        ```
    """
    _fields = []

    def __init__(self, *args, **kwargs):
        if len(args) > len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))

        # Set all of the positional arguments
        for name, value in zip(self._fields, args):
            setattr(self, name, value)

        # Set the remaining keyword arguments
        for name in self._fields[len(args):]:
            setattr(self, name, kwargs.pop(name))

        # Check for any remaining unknown arguments
        if kwargs:
            raise TypeError('Invalid argument(s): {}'.format(','.join(kwargs)))


class Bacterial(Struct):
    _fields = ['x', 'y']


def _get_mu(theta, x):
    """Computes mu as a dot product of inputs x with parameters theta."""
    mu = (theta[:, 0, np.newaxis] + np.matmul(theta[:, 1, np.newaxis],
                                              x[np.newaxis, :]))

    return np.exp(mu)


def _nll(theta, data, pi, gamma_hat):
    """Negative log-likelihood."""
    theta = theta.reshape(-1, 2)
    return -_get_log_likelihood(theta, data, pi, gamma_hat)


def _get_log_likelihood(theta, data, pi, gamma_hat):
    """Get the log-likelihood."""
    log_like = 0.0

    mu = _get_mu(theta, data.x)

    # for g in G
    for g in range(theta.shape[0]):
        log_like_g = data.y.shape[0]*math.log(pi[g])
        log_like_g += data.y*np.log(mu[g]) - mu[g]
        log_like_g *= gamma_hat[g]

        log_like += np.sum(log_like_g)

    return log_like


def _poisson_pmf(x, pi, mu):
    """pmf of Poisson."""
    like = pi*scipy.stats.poisson.pmf(x, mu=mu)

    return like


def bacterial_em():
    """Bacterial data EM."""
    num_components_G = 2

    with open('Bacterial.csv', 'r') as f:
        reader = csv.reader(f)
        dataset = np.array([l for l in reader][1:])
        dataset = dataset.astype(np.int32)
    data = Bacterial(x=dataset[:, 0], y=dataset[:, 1])

    fig, axes = plt.subplots()

    axes.hist(data.y[data.x == 0], bins=84)
    axes.hist(data.y[data.x == 1], bins=84)

    # plt.show(fig)

    gamma_hat = np.zeros([num_components_G, data.y.shape[0]])

    eps = 1e-5
    dataset_log_likes = []
    log_likelihoods = []
    # Initialize parameters
    pi_hat = np.repeat(1.0/num_components_G, num_components_G)
    theta_hat = np.random.uniform(0, 1, size=[num_components_G, 2])

    mu = _get_mu(theta_hat, data.x)
    for g in range(num_components_G):
        gamma_hat[g] = _poisson_pmf(data.y, pi_hat[g], mu[g])
    gamma_hat /= np.sum(gamma_hat, axis=0)

    log_like = _get_log_likelihood(theta_hat, data, pi_hat, gamma_hat)
    print(log_like)
    log_likelihoods.append(log_like)

    while True:
        # E step. Compute responsibilities (expected value of the
        # proportion of samples drawn from a given component).
        mu = _get_mu(theta_hat, data.x)
        for g in range(num_components_G):
            gamma_hat[g] = _poisson_pmf(data.y, pi_hat[g], mu[g])
        gamma_hat /= np.sum(gamma_hat, axis=0)

        # M step.
        optimize_result = scipy.optimize.minimize(_nll,
                                                  theta_hat,
                                                  args=(data, pi_hat, gamma_hat))
        theta_hat = optimize_result.x.reshape(-1, 2)
        pi_hat = gamma_hat.mean(axis=1)
        print(pi_hat)

        # Convergence criteria.
        log_like = _get_log_likelihood(theta_hat, data, pi_hat, gamma_hat)
        print(log_like)
        log_likelihoods.append(log_like)

        delta = (log_likelihoods[-1] - log_likelihoods[-2])
        assert delta >= 0
        if abs(delta) < eps:
            break

    print(log_likelihoods)
    print(pi_hat)
    print(theta_hat)


if __name__ == '__main__':
    bacterial_em()
