from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
import numpy as np
@dataclass
class OUParams:
    alpha: float  # mean reversion parameter
    gamma: float  # asymptotic mean
    beta: float  # Brownian motion scale (standard deviation)

def estimate_OU_params(X_t: np.ndarray) -> OUParams:
    """
    Estimate OU params from OLS regression.
    - X_t is a 1D array.
    Returns instance of OUParams.
    OU process is defined by the following SDE:
    dX_t = alpha * (gamma - X_t) * dt + beta * dW_t
    where W_t is a Brownian motion.
    alpha is the mean reversion parameter (Avelaneda calls it Kappa). Represents speed of mean reversion.
        (1/alpha is the characteristic time of mean reversion)
    gamma is the asymptotic mean (Avelaneda calls it M)
    beta is the Brownian motion scale (standard deviation) (Avelaneda calls it Sigma)
    """
    y = np.diff(X_t)
    X = X_t[:-1].reshape(-1, 1)
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    # regression coeficient and constant
    alpha = -reg.coef_[0]
    gamma = reg.intercept_ / alpha
    # residuals and their standard deviation
    y_hat = reg.predict(X)
    beta = np.std(y - y_hat)
    return OUParams(alpha, gamma, beta)