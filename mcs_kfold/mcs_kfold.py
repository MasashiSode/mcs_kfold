import os
import random

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from joblib import Parallel, delayed

class MCSKFold:
    def __init__(
        self,
        n_splits: int = 5,
        max_iter: int = 100,
        shuffle_mc: bool = True,
        global_seed: int = 2020,
    ):
        n_splits, max_iter = self.__check_input(n_splits, max_iter)
        self.n_splits = n_splits
        self.max_iter = max_iter
        self.shuffle_mc = shuffle_mc
        self.global_seed = global_seed
        self.__seed_everything()

    def split(self, df: pd.DataFrame, target_cols: list, target_cols_cat_num: list = None):
        seeds = self.__initialize_seed()
        df = self.__convert_cat_var_to_int(df, target_cols)
        if target_cols_cat_num is None:
            target_cols_cat_num = self.__count_unique(df, target_cols)

        df_result = Parallel(n_jobs=-1)([delayed(self.split_one_seed)(df, target_cols, target_cols_cat_num, seed) for seed in seeds])
        df_result = pd.concat(df_result)

        best_kfold_seed = self.__extract_smallest_variance_seed_in_topk(df_result)
        indices = self.__get_best_kfold(df, target_cols, best_kfold_seed)
        return indices

    def split_one_seed(self, df: pd.DataFrame, target_cols: list, target_cols_cat_num: list = None, seed=0):
        df_result = pd.DataFrame()

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
        df_list = []
        num_list = []

        for fold, (train_idx, valid_idx) in enumerate(skf.split(df.index, df[target_cols[0]])):
            df_list.append(df.iloc[train_idx])
            num_list.append(len(train_idx))
        num_min = min(num_list)

        target_cols_cat_num_max = max(target_cols_cat_num)
        df_result.loc[str(seed), "weighted_mean"] = 0
        df_result.loc[str(seed), "weighted_var"] = 0
        for target_col_index, target_col in enumerate(target_cols[1:]):
            entropy_i_j_list = []
            for i, df_i in enumerate(df_list):
                df_i = df_i[target_cols[1:]].iloc[:num_min]
                for j, df_j in enumerate(df_list):
                    if j >= i:
                        continue
                    df_j = df_j[target_cols[1:]].iloc[:num_min]

                    entropy_i_j = entropy(
                        np.histogram(
                            df_i[target_col],
                            bins=target_cols_cat_num[target_col_index + 1],
                            density=True,
                        )[0],
                        np.histogram(
                            df_j[target_col],
                            bins=target_cols_cat_num[target_col_index + 1],
                            density=True,
                        )[0],
                    )
                    entropy_i_j_list.append(entropy_i_j)
            entropy_mean_col = np.mean(entropy_i_j_list)
            entropy_var_col = np.var(entropy_i_j_list)

            df_result.loc[str(seed), target_col] = entropy_mean_col
            df_result.loc[str(seed), f"{target_col}_var"] = entropy_var_col
            df_result.loc[str(seed), "weighted_mean"] += (
                target_cols_cat_num_max / target_cols_cat_num[target_col_index + 1]
            ) * entropy_mean_col

            df_result.loc[str(seed), "weighted_var"] += (
                target_cols_cat_num_max / target_cols_cat_num[target_col_index + 1]
            ) * entropy_var_col

        return df_result

    def __seed_everything(self):
        random.seed(self.global_seed)
        os.environ["PYTHONHASHSEED"] = str(self.global_seed)
        np.random.seed(self.global_seed)

    def __check_input(self, n_splits, max_iter):
        if n_splits <= 1:
            n_splits = 2
        if max_iter <= 1:
            max_iter = 1
        return n_splits, max_iter

    def __count_unique(self, df, target_cols):
        return df[target_cols].nunique()

    def __initialize_seed(self):
        if self.shuffle_mc:
            seeds = [random.randint(0, self.max_iter * 10) for i in range(self.max_iter)]
        else:
            seeds = range(0, self.max_iter)
        return seeds

    def __convert_cat_var_to_int(self, df, target_cols):
        lb_make = LabelEncoder()
        cat_targets = df[target_cols].columns[df[target_cols].dtypes == "object"]
        for cat_target in cat_targets:
            df[cat_target] = lb_make.fit_transform(df[cat_target])
        return df

    def __extract_smallest_variance_seed_in_topk(self, df_result):
        best_kfold_index = df_result.nsmallest(10, "weighted_mean")["weighted_var"].argmin()
        best_kfold_seed = int(df_result.nsmallest(10, "weighted_mean").index[best_kfold_index])
        return best_kfold_seed

    def __get_best_kfold(self, df, target_cols, best_kfold_seed):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=best_kfold_seed)
        indices = []
        for _, (train_idx, valid_idx) in enumerate(skf.split(df.index, df[target_cols[0]])):
            indices.append([train_idx, valid_idx])
        return indices
