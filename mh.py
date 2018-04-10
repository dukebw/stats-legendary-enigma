"""Metropolis-Hastings."""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def mh():
    """Metropolis-Hastings."""
    p = scipy.stats.multivariate_normal(mean=0, cov=1)

    plt.xlabel('Iterations')
    plt.ylabel('x')
    total_iter = 1000
    expected_mus = np.zeros([4, 10])
    acceptance_probs = np.zeros([4, 10])
    tau_squareds = [0.1, 1, 10, 100]
    for mu_i in range(expected_mus.shape[1]):
        for tau_i, tau_squared in enumerate(tau_squareds):
            x = 10
            num_accepted = 0
            all_x = []
            for _ in range(total_iter):
                all_x.append(x)

                # Sample x' drawn from q(x'|x)
                q = scipy.stats.multivariate_normal(mean=x, cov=tau_squared)
                xprime = q.rvs()

                # Compute acceptance probability alpha
                qprime = scipy.stats.multivariate_normal(mean=xprime,
                                                         cov=tau_squared)
                alpha = p.pdf(xprime)*qprime.pdf(x)/(p.pdf(x)*q.pdf(xprime))

                r = min(1, alpha)
                u = scipy.stats.uniform.rvs(0, 1)
                if u < r:
                    num_accepted += 1
                    x = xprime

            expected_mus[tau_i, mu_i] = np.mean(all_x[200:])
            acceptance_probs[tau_i, mu_i] = np.mean(num_accepted/total_iter)
            # plt.plot(all_x)
            # plt.show()

    for tau_i, tau_squared in enumerate(tau_squareds):
        print(f'tau^2: {tau_squared}')
        accept_prob = np.mean(acceptance_probs[tau_i])
        accept_std = np.std(acceptance_probs[tau_i])
        print(f'acceptance prob: {accept_prob} +- {accept_std}')
        expected_mu = np.mean(expected_mus[tau_i])
        mu_std = np.std(expected_mus[tau_i])
        print(f'expected mu: {expected_mu} +- {mu_std}')
        print()


if __name__ == '__main__':
    mh()
