# mcs_kfold

`mcs_kfold` stands for "monte carlo stratified k fold". This library attempts to achieve equal distribution of discrete variables/categorical in all folds.
Internally, the seed is changed and stratified k-fold trials are repeated to find the seed with the least entropy in the distribution of the specified variables. The greatest advantage of this method is that it can be applied to multi-dimensional targets.

## Usage

```python
from mcs_kfold import MCSKFold
mcskf = MCSKFold(n_splits=num_cv, shuffle_mc=True, max_iter=100)

for fold, (train_idx, valid_idx) in enumerate(
    mcskf.split(df=df, target_cols=["survived", "Pclass", "Sex"])
):
    .
    .
    .


```

## Install

### pip

**not yet implemented**

`pip install mcs_kfold`

### Install newest version

```sh
git clone []
cd []
pip install .
```
