from dataclasses import dataclass


@dataclass(frozen=True)
class ElasticNet:
    fit_intercept: bool = True
    l1_ratio: float = 0.5
    alphas: float = 0.1
    cv: int = 5
    max_iter: int = 50_000
    n_jobs: int = -1
    random_state: int = 42


ELASTICNET = ElasticNet()
