import numpy as np
#Hack around deprecation of np.bool in np v 1.25 and qpython 2.0 still referring to it
np.bool = np.bool_
np.str = np.str_
#Hack around deprecation of np.bool in np v 1.25 and qpython 2.0 still referring to it
from qpython import qconnection,MetaData, qtemporal
from qpython.qconnection import QConnection