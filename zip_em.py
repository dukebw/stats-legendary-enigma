"""Find the MLE of a ZIP (Zero-Inflated Poisson) distribution using Expectation
Maximization.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def _get_log_likelihood(samples, pi, lambd):
    """Get the likelihood."""
    log_like = 0.0
    for sample in samples:
        log_like += math.log(_zip_pmf(sample, pi, lambd))

    return log_like


def _zip_pmf(x, pi, lambd):
    """pmf of ZIP."""
    like = 0.0
    if x == 0:
        like += pi

    like += (1 - pi)*scipy.stats.poisson.pmf(x, mu=lambd)

    return like


def zip_em():
    """ZIP EM."""
    num_datasets = 100
    num_samples = 100
    # NOTE(brendan): ground truth values.
    pi_gt = 0.3
    lambda_gt = 3

    support_vals = [i for i in range(20)]
    support_probs = len(support_vals)*[0.0]
    for i in range(len(support_vals)):
        support_probs[i] = _zip_pmf(i, pi_gt, lambda_gt)

    random_variable = scipy.stats.rv_discrete(
        values=(support_vals, support_probs))

    samples = np.zeros((num_datasets, num_samples))
    for i in range(num_datasets):
        samples[i] = random_variable.rvs(size=num_samples)

    fig, axes = plt.subplots(nrows=1,
                             ncols=num_datasets,
                             sharey=True,
                             tight_layout=True)

    for i in range(num_datasets):
        axes[i].hist(samples[i], bins=len(support_vals))

    # plt.show(fig)

    gamma_hat = np.zeros(num_samples)

    eps = 1e-5
    dataset_log_likes = []
    thetas = np.zeros((num_datasets, 2))
    for set_i in range(num_datasets):
        log_likelihoods = []
        # Initialize parameters
        pi_hat = len(samples[set_i][samples[set_i] == 0])/num_samples
        lambda_hat = samples[set_i].mean()

        log_like = _get_log_likelihood(samples[set_i], pi_hat, lambda_hat)
        log_likelihoods.append(log_like)

        while True:
            # E step. Compute responsibilities (expected value of the proportion of
            # samples drawn from the distribution that always draws x == 0).
            for i in range(num_samples):
                if samples[set_i, i] == 0:
                    gamma_hat[i] = pi_hat/_zip_pmf(0, pi_hat, lambda_hat)
                else:
                    gamma_hat[i] = 0

            # M step.
            pi_hat = gamma_hat.mean()
            lambda_hat = ((1.0 - gamma_hat)*samples[set_i]).sum()
            lambda_hat /= (1.0 - gamma_hat).sum()

            # Convergence criteria.
            log_like = _get_log_likelihood(samples[set_i], pi_hat, lambda_hat)
            log_likelihoods.append(log_like)

            delta = (log_likelihoods[-1] - log_likelihoods[-2])
            assert delta >= 0
            if abs(delta) < eps:
                break

        dataset_log_likes.append(log_likelihoods)
        thetas[set_i, 0] = pi_hat
        thetas[set_i, 1] = lambda_hat

    theta_mean = thetas.mean(axis=0)
    theta_variance = (thetas - theta_mean)**2
    theta_variance = theta_variance.mean(axis=0)
    theta_mse = (thetas - np.array([pi_gt, lambda_gt]))**2
    theta_mse = theta_mse.mean(axis=0)
    print('pi: mean {} variance {} MSE {}'.format(theta_mean[0], theta_variance[0], theta_mse[0]))
    print('lambda: mean {} variance {} MSE {}'.format(theta_mean[1], theta_variance[1], theta_mse[1]))


if __name__ == '__main__':
    zip_em()
