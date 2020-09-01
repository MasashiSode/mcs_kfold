import os
import random
from collections import namedtuple

import numpy as np

from mcs_kfold import MCSKFold


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(2020)


def test_mcs_kfold(num_cv):
    mcskf = MCSKFold(n_splits=num_cv, max_iter=100)
    assert isinstance(mcskf, MCSKFold)


def test_split():
    pass
