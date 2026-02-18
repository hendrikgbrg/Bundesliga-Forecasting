from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ElasticNet:
    fit_intercept: bool = True
    l1_ratio: tuple = (0.5, 0.7, 0.9, 0.95, 0.99)
    alphas: np.ndarray = field(default_factory=lambda: np.logspace(-2, 1, 100))
    cv: int = 5
    max_iter: int = 50_000
    n_jobs: int = -1


ELASTICNET = ElasticNet()
