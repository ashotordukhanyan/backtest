from dataclasses import dataclass
from typing import List, Tuple
@dataclass
class SymStatParams:
    FACTOR_ETFS : List[str] # ETFS to consider for hedging
    HEDGE_ETFS_PER_NAME : int # hom many etfs to consider for hedging each individual stock
    BETA_R_ALPHA : float # Alpha for ridge regression that determines the hedge weights
    BETA_WINDOW_SIZE : int = 24 # in months - 2
    BETA_MIN_OBS : int = 12 # in months minimum number of observations for beta calculation
    BETA_LIMITS : Tuple[float,float] = (-5.,5.)  # floor/cap for calculated betas


SYMSTAT_PARAMS = SymStatParams(FACTOR_ETFS = 'SPY RTH XLF XLY XLP SMH XLI XLU XLV KRE IYR OIH XLK IYT XLE QQQ'.split(),
                                             HEDGE_ETFS_PER_NAME = 1,
                                             BETA_R_ALPHA = 0.0001)

