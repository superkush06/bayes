"""Metropolis-Hastings (random walk) for arbitrary log-posteriors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MHResult:
    samples: np.ndarray         # (n_samples, dim)
    log_posts: np.ndarray       # (n_samples,)
    accept_rate: float


def metropolis_hastings(log_post, x0: np.ndarray, *, n_samples: int,
                        step_size: float = 0.1, seed: int = 0,
                        burn_in: int = 0) -> MHResult:
    """Random-walk Metropolis-Hastings on R^d.

    log_post(x) -> log p(x). We propose x' = x + step_size * z, z ~ N(0, I),
    and accept with probability min(1, exp(log_post(x') - log_post(x))).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).reshape(-1)
    d = x.shape[0]
    total_iter = n_samples + burn_in
    samples = np.zeros((n_samples, d))
    log_posts = np.zeros(n_samples)
    n_accept = 0
    lp = log_post(x)
    kept = 0
    for k in range(total_iter):
        proposal = x + step_size * rng.standard_normal(d)
        lp_new = log_post(proposal)
        log_alpha = lp_new - lp
        if np.log(rng.random() + 1e-300) < log_alpha:
            x = proposal
            lp = lp_new
            if k >= burn_in:
                n_accept += 1
        if k >= burn_in:
            samples[kept] = x
            log_posts[kept] = lp
            kept += 1
    return MHResult(samples=samples, log_posts=log_posts,
                    accept_rate=n_accept / max(1, n_samples))
