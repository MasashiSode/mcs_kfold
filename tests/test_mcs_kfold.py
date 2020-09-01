import pandas as pd

from mcs_kfold import MCSKFold
import pytest

SEED = 2020


def make_param_mcs_kfold():
    test_params = []
    for num_cv in range(0, 15):
        for max_iter in [0, 100, 10]:
            for shuffle_mc in [True, False]:
                test_params.append((num_cv, max_iter, shuffle_mc))
    return test_params


def make_param_split():
    test_params = []
    for num_cv in range(0, 15):
        for shuffle_mc in [True, False]:
            test_params.append((num_cv, shuffle_mc))
    return test_params


@pytest.mark.parametrize("num_cv, max_iter, shuffle_mc", make_param_mcs_kfold())
def test_mcs_kfold(num_cv, max_iter, shuffle_mc):
    mcskf = MCSKFold(n_splits=num_cv, max_iter=max_iter, shuffle_mc=shuffle_mc, global_seed=SEED)
    assert isinstance(mcskf, MCSKFold)


@pytest.mark.parametrize("num_cv, shuffle_mc", make_param_split())
def test_split(num_cv, shuffle_mc):
    df = pd.read_csv("tests/test_data/train_titanic.csv")
    mcskf = MCSKFold(n_splits=num_cv, max_iter=1, shuffle_mc=shuffle_mc, global_seed=SEED)
    indices = mcskf.split(df=df, target_cols=["Survived", "Pclass", "Sex"])
    assert isinstance(indices, list)


if __name__ == "__main__":
    test_split()
