"""Poisson distribution"""

from collections import namedtuple

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.special import digamma, gamma, polygamma

from xgboost_distribution.distributions.base import BaseDistribution
from xgboost_distribution.distributions.utils import (
    check_all_ge_zero,
    safe_exp,
)

Params = namedtuple("Params", ("alpha", "beta"))


class Gamma(BaseDistribution):
    """
    alpha is shape parameter and beta rate parameter.
    """
    @property
    def params(self):
        return Params._fields

    def check_target(self, y):
        check_all_ge_zero(y)

    def gradient_and_hessian(self, y, params, natural_gradient=False):
        """Gradient and diagonal hessian
        
        differentiate exp (alpha) * beta - log(gamma(exp(alpha))) + exp(alpha)*log(y) - exp(beta)*y wrt. alpha
        differentiate exp(alpha) * (beta + log(y) - digamma(exp(alplha))) wrt alpha
        """
        _, beta_prime = params[:, 0], params[:, 1]
        alpha, beta = self.predict(params)
        grad = np.zeros(shape=(len(y), 2), dtype="float32")
        digamma_alpha = digamma(alpha)
        grad[:, 0] = - alpha * (beta_prime - digamma_alpha + np.log(y))
        grad[:, 1] = - alpha + beta * y
        # print(np.average(grad, 0))

        hess = np.zeros(shape=(len(y), 2), dtype="float32")
        hess[:, 0] = - grad[:, 0] + alpha * alpha * polygamma(1, alpha)
        # hess[:, 0, 1] = alpha
        # hess[:, 1, 0] = alpha
        hess[:, 1] = + beta * y
        # print(np.average(hess, 0))
        return grad, hess

    def loss(self, y, params):
        (alpha,beta) = self.predict(params)
        # print(-np.average(gamma_dist.logpdf(y, alpha, beta)))
        return "Gamma-NLL", -gamma_dist.logpdf(y, alpha, beta)

    def predict(self, params):
        alpha_prime, beta_prime = params[:, 0], params[:, 1]
        return Params(alpha=safe_exp(alpha_prime), beta=safe_exp(beta_prime))

    def starting_params(self, y):
        return Params(alpha=2, beta=2)
