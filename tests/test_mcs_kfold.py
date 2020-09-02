import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from mcs_kfold import MCSKFold

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


@pytest.mark.parametrize("num_cv, shuffle_mc", make_param_split())
def test_stratified_split(num_cv, shuffle_mc):
    target_cols = ["Survived", "Pclass", "Sex"]
    df = pd.read_csv("tests/test_data/train_titanic.csv")
    mcskf = MCSKFold(n_splits=num_cv, max_iter=1, shuffle_mc=shuffle_mc, global_seed=SEED)
    indices = mcskf.split(df=df, target_cols=target_cols)

    tr_ = {col: np.zeros((len(indices), len(df[col].unique()))) for col in target_cols}
    val_ = {col: np.zeros((len(indices), len(df[col].unique()))) for col in target_cols}
    for fold, (train_index, valid_index) in enumerate(indices):
        for col in target_cols:
            tr_[col][fold] = df.iloc[train_index][col].value_counts(normalize=True)
            val_[col][fold] = df.iloc[valid_index][col].value_counts(normalize=True)
    assert_threshold = 0.1
    for col in target_cols:
        tr_std_per_label = tr_[col].std(axis=0)
        val_std_per_label = val_[col].std(axis=0)
        npt.assert_array_less(tr_std_per_label, assert_threshold)
        npt.assert_array_less(val_std_per_label, assert_threshold)


if __name__ == "__main__":
    test_split()
