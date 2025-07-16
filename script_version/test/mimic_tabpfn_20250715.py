# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np

import time
import datetime


# import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm

from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_validate,
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


from sklearn.linear_model import LogisticRegression

#from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier

from sklearn.neural_network import MLPClassifier

import warnings

warnings.filterwarnings("ignore")


#from tabpfn_extensions.rf_pfn import RandomForestTabPFNClassifier

#from tabpfn_extensions import TabPFNClassifier

#from tabicl import TabICLClassifier

from joblib import Parallel, delayed

# import pickle
# import timeit

#import torch

#if not torch.cuda.is_available():
#    raise SystemError(
#        "GPU device not found. For fast training, please enable GPU. See section above for instructions."
#    )

NJOBS = 1
RANDOM_STATE = 42
N_TRAINING_SAMPLE = 4000



mimic_3 = pd.read_csv("mimic_3_processed.csv")
mimic_4 = pd.read_csv("mimic_4_processed.csv")


cont_features = [
    "age",
    "heartrate_max",
    "heartrate_min",
    "sysbp_max",
    "sysbp_min",
    "tempc_max",
    "tempc_min",
    "urineoutput",
    "bun_min",
    "bun_max",
    "wbc_min",
    "wbc_max",
    "potassium_min",
    "potassium_max",
    "sodium_min",
    "sodium_max",
    "bicarbonate_min",
    "bicarbonate_max",
    "mingcs",
    #    'pao2fio2_vent_min', 'bilirubin_min', 'bilirubin_max',
]

cat_features = ["aids", "hem", "mets", "admissiontype"]

features = cont_features + cat_features

outcome = ["hospital_mortality"]

socio_demographic = ["insurance", "marital_status", "ethnicity", "language", "gender"]

X = mimic_3[features]
y = mimic_3["hospital_mortality"]

y_proxy = mimic_3["icustay_expire_flag"]

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)


def create_directory(directory_name):
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")


def process_df(df, n_folds=n_folds):
    df_true_prob = (
        df["Prob true"]
        .explode()
        .groupby(level=0)
        .apply(lambda x: pd.Series(x.values))
        .unstack()
    )
    df_true_prob.columns = ["Prob_true_fold_" + str(i + 1) for i in range(n_folds)]
    df_pred_prob = (
        df["Prob pred"]
        .explode()
        .groupby(level=0)
        .apply(lambda x: pd.Series(x.values))
        .unstack()
    )
    df_pred_prob.columns = ["Prob_pred_fold_" + str(i + 1) for i in range(n_folds)]
    df = df.drop(["Prob true", "Prob pred"], axis=1)
    df = pd.concat([df, df_true_prob, df_pred_prob], axis=1)
    return df


date = datetime.datetime.now().date()

def save_results(
    results,
    directory,
    n_folds,
    columns=["Model", "Noise level", "AUC", "Brier score", "Prob true", "Prob pred", "Train fit time", "Test pred time"],
    test_name="MEASUREMENT NOISE",
    date=date,
):
    df_results = pd.DataFrame(results, columns=columns)
    df_results = process_df(df_results, n_folds)
    df_results.to_csv(directory + test_name + str(date) + ".csv", index=False)
    print("\n ======= \n")
    print(f"k = {test_name}")
    print(df_results.head())
    print("\n ======= \n")
    print(test_name, " OVER")


def predict_proba_batched(model, X, batch_size: int = 32_000):
    """
    Work around the CUDA 65 535-block limit in TabPFN’s SDPA kernel
    by splitting any large matrix into manageable chunks.
    Returns the class-1 probabilities concatenated in order.
    """
    if len(X) <= batch_size:  # fast path
        return model.predict_proba(X)[:, 1]

    out = []
    for start in range(0, len(X), batch_size):
        out.append(model.predict_proba(X.iloc[start : start + batch_size])[:, 1])
    return np.concatenate(out)


# MODELS

continuous_transformer_1 = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

continuous_transformer_2 = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])


categorical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
)

preprocessor_1 = ColumnTransformer(
    transformers=[
        ("cont", continuous_transformer_1, cont_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

preprocessor_2 = ColumnTransformer(
    transformers=[
        ("cont", continuous_transformer_2, cont_features),
        ("cat", categorical_transformer, cat_features),
    ]
)


#clf_base = TabPFNClassifier(
#    ignore_pretraining_limits=True,
#    inference_config={
#        "SUBSAMPLE_SAMPLES": 10000
#    },  # Needs to be set low so that not OOM on fitting intermediate nodes
#)


#tabpfn_tree_clf = RandomForestTabPFNClassifier(
#    tabpfn=clf_base,
#    verbose=0,
#    max_predict_time=60,  # Will fit for one minute
#    fit_nodes=True,  # Wheather or not to fit intermediate nodes
#    adaptive_tree=True,  # Whather or not to validate if adding a leaf helps or not
#    show_progress=True,
#)

models = {
    # Linear method
    "Logistic": Pipeline([("preprocessor", preprocessor_1), ("lr", LogisticRegression(penalty=None))]),
#    "LASSO": Pipeline([("preprocessor", preprocessor_1),("lr", LogisticRegression(penalty="l1", solver="liblinear", C=0.005)),]),
#    "Ridge": Pipeline([("preprocessor", preprocessor_1),("lr", LogisticRegression(penalty="l2", C=0.005)),]),
    # Tree-based methods
#    "Random Forest": RandomForestClassifier(),
    ## Boosting
#    "Gradient Boosting": Pipeline([("preprocessor", preprocessor_2), ("gb", GradientBoostingClassifier())]),
#    "XGBoost": XGBClassifier(),
#    "LightGBM": LGBMClassifier(),
#    "CatBoost": CatBoostClassifier(verbose=0),
    # Deep learning
#    "MLP": Pipeline([("preprocessor", preprocessor_1), ("mlp", MLPClassifier(max_iter=500))]),
    ## Transformers
#    "TabPFN": TabPFNClassifier(),
#    "TabPFN RF": tabpfn_tree_clf,  ## for more than 10k (training) samples
#    "TabICL": TabICLClassifier(),
}

models_name = list(models.keys())

n_models = len(models_name)

create_directory("results")

"""
score_crossvalidation = Parallel(n_jobs=NJOBS, verbose=0)(delayed(cross_val_score)(model, X, y, scoring='roc_auc', cv=n_folds)
                                          for model in models.values()
                                         )

score_crossvalidation_df = pd.DataFrame(
    score_crossvalidation, index=models_name, columns=["fold_"+str(i+1) for i in np.arange(n_folds)]
)

print("===== CROSSVAL =====")
print(score_crossvalidation_df.head())
score_crossvalidation_df.to_csv('results/result_crossvalidation.csv')


"""

"""## Robustness tests


### Label noise
"""


random_label_noise_levels = np.linspace(0, 1, 11)
targeted_label_noise_levels = np.linspace(0, 1, 10, endpoint=False)


def add_label_noise(y_noisy, noise_level):
    idx_change_outcome = np.random.rand(len(y_noisy)) < noise_level
    y_noisy[idx_change_outcome] = 1 - y_noisy

    return y_noisy


def label_noise(
    model_name, model, noise_level, train_idx, val_idx, noise_type="random"
):

    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]
    y_proxy_train = y_proxy.loc[train_idx]

    ## limit training to 10k samples
    X_train  = X_train.sample(N_TRAINING_SAMPLE, random_state=RANDOM_STATE) 
    y_train = y_train.sample(N_TRAINING_SAMPLE, random_state=RANDOM_STATE)
    y_proxy_train = y_proxy_train.sample(N_TRAINING_SAMPLE, random_state=RANDOM_STATE)
    
    y_train_noisy = y_train.copy()

    if noise_type == "random":
        y_train_noisy = add_label_noise(y_train_noisy, noise_level)
    elif noise_type == "0to1":
        y_train_noisy[y_train_noisy == 0] = add_label_noise(
            y_train_noisy[y_train_noisy == 0], noise_level
        )
    elif noise_type == "1to0":
        y_train_noisy[y_train_noisy == 1] = add_label_noise(
            y_train_noisy[y_train_noisy == 1], noise_level
        )
    elif noise_type == "conditional":
        age_train_perc = X_train.age.rank(pct=True)
        swap_by_age = np.random.binomial(
            1, p=age_train_perc**noise_level, size=len(age_train_perc)
        )
        y_train_noisy = y_train_noisy * (swap_by_age) + (1 - y_train_noisy) * (
            1 - swap_by_age
        )
    elif noise_type == "proxy":
        y_train_noisy = y_train_noisy * (1 - noise_level) + y_proxy_train * noise_level

    start = time.time()
    model.fit(X_train, y_train_noisy)
    end = time.time()
    train_fit_time = end - start

    start = time.time()
    y_pred = predict_proba_batched(model, X_val)
    end = time.time()
    test_pred_time = end - start

    prob_true, prob_pred = calibration_curve(y_val, y_pred)

    return (
        model_name,
        noise_level,
        roc_auc_score(y_score=y_pred, y_true=y_val),
        brier_score_loss(y_true=y_val, y_proba=y_pred),
        prob_true,
        prob_pred,
        train_fit_time,
        test_pred_time,
    )


# Safeguard
directory = "results/label_noise/"
create_directory(directory)


results_random_label_noise = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(label_noise)(
        model_name, model, noise_level, train_idx, val_idx, noise_type="random"
    )
    for model_name, model in models.items()
    for noise_level in random_label_noise_levels
    for train_idx, val_idx in kf.split(X)
)

save_results(
    results_random_label_noise, directory, n_folds, test_name="RANDOM_LABEL_NOISE"
)


results_label_noise_01 = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(label_noise)(
        model_name, model, noise_level, train_idx, val_idx, noise_type="0to1"
    )
    for model_name, model in models.items()
    for noise_level in targeted_label_noise_levels
    for train_idx, val_idx in kf.split(X)
)

save_results(results_label_noise_01, directory, n_folds, test_name="01_LABEL_NOISE")


results_label_noise_10 = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(label_noise)(
        model_name, model, noise_level, train_idx, val_idx, noise_type="1to0"
    )
    for model_name, model in models.items()
    for noise_level in targeted_label_noise_levels
    for train_idx, val_idx in kf.split(X)
)
save_results(results_label_noise_10, directory, n_folds, test_name="10_LABEL_NOISE")

results_label_noise_age = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(label_noise)(
        model_name, model, noise_level, train_idx, val_idx, noise_type="conditional"
    )
    for model_name, model in models.items()
    for noise_level in random_label_noise_levels
    for train_idx, val_idx in kf.split(X)
)

save_results(results_label_noise_10, directory, n_folds, test_name="AGE_LABEL_NOISE")


results_label_noise_proxy = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(label_noise)(
        model_name, model, noise_level, train_idx, val_idx, noise_type="proxy"
    )
    for model_name, model in models.items()
    for noise_level in [0, 1]
    for train_idx, val_idx in kf.split(X)
)


save_results(results_label_noise_10, directory, n_folds, test_name="PROXY_LABEL_NOISE")


print("LABEL NOISE OVER")

"""### Measurement noise
---
"""

input_noise_level = np.linspace(0, 1, 11)


def add_measurement_noise(X, noise_level, feature_type="cont & cat"):

    X_noisy = X.copy()
    size = X_noisy.shape

    if "cont" in feature_type:
        for j in cont_features:
            std_j = X_noisy[j].std()
            noise_j = np.random.normal(scale=std_j * noise_level, size=size[0])
            X_noisy[j] = X_noisy[j] + noise_j

    if "cat" in feature_type:
        for j in cat_features:
            max_xj = X_noisy[j].max()
            min_xj = X_noisy[j].min()
            mask = np.random.binomial(1, noise_level, size=size[0])
            noise = np.random.randint(
                0, max_xj, size=size[0]
            )  ## only works for consecutive integers
            noise[noise == X_noisy[j].values] = (
                noise[noise == X_noisy[j].values] + 1
            ) % max_xj
            X_noisy[j] = np.where(mask, noise, X_noisy[j])

    return X_noisy


def input_noise(
    model_name,
    model,
    noise_level,
    train_idx,
    val_idx,
    which_set="Train",
    feature_type="cont & cat",
):

    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]

    ## limit training to 10k samples
    X_train  = X_train.sample(N_TRAINING_SAMPLE, random_state=RANDOM_STATE) 
    y_train = y_train.sample(N_TRAINING_SAMPLE, random_state=RANDOM_STATE)
    

    if which_set == "Train":
        X_train = add_measurement_noise(X_train, noise_level, feature_type)
    if which_set == "Val":
        X_val = add_measurement_noise(X_val, noise_level, feature_type)
    if which_set == "Train_Val":
        X_train = add_measurement_noise(X_train, noise_level, feature_type)
        X_val = add_measurement_noise(X_val, noise_level, feature_type)

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    train_fit_time = end - start

    start = time.time()
    y_pred = predict_proba_batched(model, X_val)
    end = time.time()
    test_pred_time = end - start

    prob_true, prob_pred = calibration_curve(y_val, y_pred)

    return (
        model_name,
        noise_level,
        roc_auc_score(y_score=y_pred, y_true=y_val),
        brier_score_loss(y_true=y_val, y_proba=y_pred),
        prob_true,
        prob_pred,
        train_fit_time,
        test_pred_time,
    )


directory = "results/input_noise/"
create_directory(directory)

results_input_noise_train = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(model_name, model, noise_level, train_idx, val_idx)
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for train_idx, val_idx in kf.split(X)
)

save_results(
    results_input_noise_train, directory, n_folds, test_name="INPUT_NOISE_TRAIN"
)


results_input_noise_val = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(
        model_name, model, noise_level, train_idx, val_idx, which_set="Val"
    )
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for train_idx, val_idx in kf.split(X)
)

save_results(results_input_noise_val, directory, n_folds, test_name="INPUT_NOISE_VAL")

results_input_noise_all = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(
        model_name, model, noise_level, train_idx, val_idx, which_set="Train_Val"
    )
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for train_idx, val_idx in kf.split(X)
)

save_results(results_input_noise_all, directory, n_folds, test_name="INPUT_NOISE_ALL")

results_input_noise_train_cont = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(
        model_name, model, noise_level, train_idx, val_idx, feature_type="cont"
    )
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for train_idx, val_idx in kf.split(X)
)

save_results(
    results_input_noise_train_cont,
    directory,
    n_folds,
    test_name="INPUT_NOISE_TRAIN_CONT",
)

results_input_noise_val_cont = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(
        model_name,
        model,
        noise_level,
        train_idx,
        val_idx,
        which_set="Val",
        feature_type="cont",
    )
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for train_idx, val_idx in kf.split(X)
)

save_results(
    results_input_noise_val_cont, directory, n_folds, test_name="INPUT_NOISE_VAL_CONT"
)

results_input_noise_all_cont = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(
        model_name,
        model,
        noise_level,
        train_idx,
        val_idx,
        which_set="Train_Val",
        feature_type="cont",
    )
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for train_idx, val_idx in kf.split(X)
)

save_results(
    results_input_noise_all_cont, directory, n_folds, test_name="INPUT_NOISE_ALL_CONT"
)

results_input_noise_train_cat = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(
        model_name, model, noise_level, train_idx, val_idx, feature_type="cat"
    )
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for train_idx, val_idx in kf.split(X)
)

save_results(
    results_input_noise_train_cat, directory, n_folds, test_name="INPUT_NOISE_TRAIN_CAT"
)

results_input_noise_val_cat = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(
        model_name,
        model,
        noise_level,
        train_idx,
        val_idx,
        which_set="Val",
        feature_type="cat",
    )
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for train_idx, val_idx in kf.split(X)
)

save_results(
    results_input_noise_val_cat, directory, n_folds, test_name="INPUT_NOISE_VAL_CAT"
)

results_input_noise_all_cat = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(
        model_name,
        model,
        noise_level,
        train_idx,
        val_idx,
        which_set="Train_Val",
        feature_type="cat",
    )
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for train_idx, val_idx in kf.split(X)
)

save_results(
    results_input_noise_all_cat, directory, n_folds, test_name="INPUT_NOISE_ALL_CAT"
)


print("MEASUREMENT NOISE OVER")

"""### Imbalance data

---

"""

imbalance_ratio = np.linspace(1, 0, 10, endpoint=False)


def imbalance_data(model_name, model, imbalance_ratio, train_idx, val_idx):

    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]



    X_train_negative = X_train.loc[y_train == 0].sample(frac=imbalance_ratio)
    y_train_negative = y_train.loc[X_train_negative.index]
    X_train_balanced = pd.concat([X_train_negative, X_train.loc[y_train == 1]])
    y_train_balanced = pd.concat([y_train_negative, y_train.loc[y_train == 1]])

    ## limit training to 10k samples
    X_train_balanced  = X_train_balanced.sample(N_TRAINING_SAMPLE, random_state=RANDOM_STATE) 
    y_train_balanced = y_train_balanced.sample(N_TRAINING_SAMPLE, random_state=RANDOM_STATE)


    start = time.time()
    model.fit(X_train_balanced, y_train_balanced)
    end = time.time()
    train_fit_time = end - start

    start = time.time()
    y_pred = predict_proba_batched(model, X_val)
    end = time.time()
    test_pred_time = end - start

    prob_true, prob_pred = calibration_curve(y_val, y_pred)

    return (
        model_name,
        imbalance_ratio,
        roc_auc_score(y_score=y_pred, y_true=y_val),
        brier_score_loss(y_true=y_val, y_proba=y_pred),
        prob_true,
        prob_pred,
        train_fit_time,
        test_pred_time,
    )


results_imbalance_data = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(imbalance_data)(model_name, model, ratio, train_idx, val_idx)
    for model_name, model in models.items()
    for ratio in imbalance_ratio
    for train_idx, val_idx in kf.split(X)
)

directory = "results/imbalance_data/"
create_directory(directory)

save_results(results_imbalance_data, directory, n_folds, test_name="IMBALANCED_DATA")


print("IMBALANCE NOISE OVER")


"""### Missing data
---

"""

directory = "results/missing_data/"
create_directory(directory)


def add_missingness(X, noise_level, mechanism="MCAR", q=0.7, prop_cond_features=0.5):
    X_noisy = X.copy()
    size = X_noisy.shape

    if mechanism == "MCAR":
        nan_mask = np.random.rand(*size) < noise_level
        X_noisy = X_noisy.mask(
            nan_mask, np.nan
        )  ## .mask will replace cells containing True with NaNs

    elif mechanism == "MAR":
        nan_mask = np.random.rand(*size) < noise_level
        cond_features = np.random.choice(
            X.columns, size=int(len(X.columns) * prop_cond_features), replace=False
        )
        missing_v_features = [c for c in X.columns if c not in cond_features]
        X_quantile = X[cond_features].rank(pct=True)
        nan_mask_mar = (X_quantile > q) & nan_mask
        X_noisy = X_noisy.mask(nan_mask_mar, np.nan)

    elif mechanism == "MNAR":
        X_quantile = X.rank(pct=True)  ## returns percentile
        nan_mask = (np.random.rand(*size) < noise_level) & (X_quantile > q)
        X_noisy = X_noisy.mask(nan_mask, np.nan)

    return X_noisy


def missing_data(
    model_name,
    model,
    noise_level,
    train_idx,
    val_idx,
    which_set="Train",
    mechanism="MNAR",
):

    X_train, X_val = (
        X[cont_features + cat_features].loc[train_idx],
        X[cont_features + cat_features].loc[val_idx],
    )
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]

    ## limit training to 10k samples
    X_train  = X_train.sample(N_TRAINING_SAMPLE, random_state=RANDOM_STATE) 
    y_train = y_train.sample(N_TRAINING_SAMPLE, random_state=RANDOM_STATE)


    if which_set == "Train":
        X_train = add_missingness(X_train, noise_level)
    if which_set == "Val":
        X_val = add_missingness(X_val, noise_level)
    if which_set == "Train_Val":
        X_train = add_missingness(X_train, noise_level)
        X_val = add_missingness(X_val, noise_level)

    ## Naîve imputation with mean. Perhaps explore other imputation methods? Embedded imputation methods for methods such as gradient boosting

    mean_values_tr = X_train[cont_features].mean()
    most_frequent_values_tr = X_train[cat_features].mode().loc[0]

    if model_name in ["LogisticRegression", "LASSO", "Ridge", "Gradient Boosting", "MLP"]:
        X_train[cont_features] = X_train[cont_features].fillna(mean_values_tr)
        X_val[cont_features] = X_val[cont_features].fillna(mean_values_tr)

        X_train[cat_features] = X_train[cat_features].fillna(most_frequent_values_tr)
        X_val[cat_features] = X_val[cat_features].fillna(most_frequent_values_tr)

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    train_fit_time = end - start

    start = time.time()
    y_pred = predict_proba_batched(model, X_val)
    end = time.time()
    test_pred_time = end - start

    prob_true, prob_pred = calibration_curve(y_val, y_pred)
    return (
        model_name,
        noise_level,
        roc_auc_score(y_score=y_pred, y_true=y_val),
        brier_score_loss(y_true=y_val, y_proba=y_pred),
        prob_true,
        prob_pred,
        train_fit_time,
        test_pred_time,
    )


missing_ratio = np.linspace(0, 1, 10, endpoint=False)

results_MCAR_train = Parallel(n_jobs=NJOBS, verbose=1)(
    delayed(missing_data)(
        model_name, model, ratio, train_idx, val_idx, which_set="Train"
    )
    for model_name, model in models.items()
    for ratio in missing_ratio
    for train_idx, val_idx in kf.split(X)
)

save_results(results_MCAR_train, directory, n_folds, test_name="MCAR_TRAIN")

results_MCAR_val = Parallel(n_jobs=NJOBS, verbose=1)(
    delayed(missing_data)(model_name, model, ratio, train_idx, val_idx, which_set="Val")
    for model_name, model in models.items()
    for ratio in missing_ratio
    for train_idx, val_idx in kf.split(X)
)

save_results(results_MCAR_val, directory, n_folds, test_name="MCAR_VAL")

results_MCAR_all = Parallel(n_jobs=NJOBS, verbose=1)(
    delayed(missing_data)(
        model_name, model, ratio, train_idx, val_idx, which_set="Train_Val"
    )
    for model_name, model in models.items()
    for ratio in missing_ratio
    for train_idx, val_idx in kf.split(X)
)

save_results(results_MCAR_all, directory, n_folds, test_name="MCAR_ALL")

"""
results_missing_data_train_mnar = Parallel(n_jobs=NJOBS, verbose=1)(delayed(missing_data)(model_name, model, ratio, train_idx, val_idx, which_set='Train', mechanism='MNAR') 
                                          for model_name, model in models.items()
                                          for ratio in missing_ratio
                                          for train_idx, val_idx in kf.split(X)
                                         )

save_results(results_imbalance_data, directory, n_folds, test_name="IMBALANCED_DATA")


results_missing_data_val_mnar = Parallel(n_jobs=NJOBS, verbose=1)(delayed(missing_data)(model_name, model, ratio, train_idx, val_idx, which_set='Val', mechanism='MNAR') 
                                          for model_name, model in models.items()
                                          for ratio in missing_ratio
                                          for train_idx, val_idx in kf.split(X)
                                         )

save_results(results_imbalance_data, directory, n_folds, test_name="IMBALANCED_DATA")


results_missing_data_all_mnar = Parallel(n_jobs=NJOBS, verbose=1)(delayed(missing_data)(model_name, model, ratio, train_idx, val_idx, which_set='Train_Val', mechanism='MNAR') 
                                          for model_name, model in models.items()
                                          for ratio in missing_ratio
                                          for train_idx, val_idx in kf.split(X)
                                         )

save_results(results_imbalance_data, directory, n_folds, test_name="IMBALANCED_DATA")

"""

print("MISSING DATA OVER")


"""### Feature shuffling"""

shuffle_ratio = np.linspace(0, 1, 11, endpoint=True)
directory = "results/feature_shuffle/"
create_directory(directory)


def shuffle_features(X, prop=0.5, feat_to_shuffle=None):
    X_noisy = X.copy()
    size = X_noisy.shape

    n_feat_to_shuffle = int(size[1] * prop)
    if feat_to_shuffle is None:
        feat_to_shuffle = np.random.choice(
            X_noisy.columns, size=n_feat_to_shuffle, replace=False
        )
    X_noisy[feat_to_shuffle] = X_noisy[feat_to_shuffle].apply(np.random.permutation)

    return X_noisy, feat_to_shuffle


def permutation_features(
    model_name, model, noise_level, train_idx, val_idx, which_set="Train"
):

    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]

    ## limit training to 10k samples
    X_train  = X_train.sample(N_TRAINING_SAMPLE, random_state=RANDOM_STATE) 
    y_train = y_train.sample(N_TRAINING_SAMPLE, random_state=RANDOM_STATE)

    if which_set == "Train":
        X_train, _ = shuffle_features(X_train, noise_level)
    if which_set == "Val":
        X_val, _ = shuffle_features(X_val, noise_level)
    if which_set == "Train_Val":
        X_train, feat_to_shuffle = shuffle_features(X_train, noise_level)
        X_val, _ = shuffle_features(X_val, noise_level, feat_to_shuffle=feat_to_shuffle)

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    train_fit_time = end - start

    start = time.time()
    y_pred = predict_proba_batched(model, X_val)
    end = time.time()
    test_pred_time = end - start

    prob_true, prob_pred = calibration_curve(y_val, y_pred)

    return (
        model_name,
        noise_level,
        roc_auc_score(y_score=y_pred, y_true=y_val),
        brier_score_loss(y_true=y_val, y_proba=y_pred),
        prob_true,
        prob_pred,
        train_fit_time,
        test_pred_time,
    )


results_shuffled_data_train = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(permutation_features)(
        model_name, model, ratio, train_idx, val_idx, which_set="Train"
    )
    for model_name, model in models.items()
    for ratio in shuffle_ratio
    for train_idx, val_idx in kf.split(X)
)

save_results(
    results_shuffled_data_train, directory, n_folds, test_name="SHUFFLED_TRAIN_DATA"
)

results_shuffled_data_val = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(permutation_features)(
        model_name, model, ratio, train_idx, val_idx, which_set="Val"
    )
    for model_name, model in models.items()
    for ratio in shuffle_ratio
    for train_idx, val_idx in kf.split(X)
)

save_results(
    results_shuffled_data_val, directory, n_folds, test_name="SHUFFLED_VAL_DATA"
)

results_shuffled_data_all = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(permutation_features)(
        model_name, model, ratio, train_idx, val_idx, which_set="Train_Val"
    )
    for model_name, model in models.items()
    for ratio in shuffle_ratio
    for train_idx, val_idx in kf.split(X)
)

save_results(
    results_shuffled_data_all, directory, n_folds, test_name="SHUFFLED_ALL_DATA"
)


print("SHUFFLE NOISE OVER")

"""### Subgroup analysis

Here we consider different types of subgroup analysis: both within MIMIC-III and across MIMIC-IV. <br> For each analysis we consider either stratifying on one of the included feature (e.g., age) in the model or on a external variable (e.g., gender, icu unit).  
"""

directory = "results/subgroups/"
create_directory(directory)

"""#### MIMIC-III

**By gender**
"""


def subgroup_analysis(model_name, model, train_idx, val_idx, stratify_on="gender"):

    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]

    ## limit training to 10k samples
    X_train  = X_train.sample(N_TRAINING_SAMPLE, random_state=RANDOM_STATE) 
    y_train = y_train.sample(N_TRAINING_SAMPLE, random_state=RANDOM_STATE)

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    train_fit_time = end - start

    stratified_perf = []

    for cat in mimic_3[stratify_on].dropna().unique():
        X_val_strat = X_val.loc[mimic_3[stratify_on] == cat]
        y_val_strat = y_val.loc[X_val_strat.index]
        start = time.time()
        y_pred = predict_proba_batched(model, X_val_strat)
        end = time.time()
        test_pred_time = end - start

        prob_true, prob_pred = calibration_curve(y_val_strat, y_pred)
        stratified_perf.append(
            [
                model_name,
                stratify_on,
                cat,
                roc_auc_score(y_score=y_pred, y_true=y_val_strat),
                brier_score_loss(y_true=y_val_strat, y_proba=y_pred),
                prob_true,
                prob_pred,
                train_fit_time,
                test_pred_time,
            ]
        )

    return stratified_perf


results_subgroup_m3_gender = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(subgroup_analysis)(model_name, model, train_idx, val_idx, variable)
    for model_name, model in models.items()
    for variable in ["gender"]
    for train_idx, val_idx in kf.split(X)
)
results_subgroup_m3_gender = [x for xs in results_subgroup_m3_gender for x in xs]

save_results(
    results_subgroup_m3_gender,
    directory,
    n_folds,
    columns=[
        "Model",
        "Variable",
        "Category",
        "AUC",
        "Brier score",
        "Prob true",
        "Prob pred",
        "Train fit time",
        "Test pred time",
    ],
    test_name="SUBGROUP_GENDER",
)


results_subgroup_m3_agegroup = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(subgroup_analysis)(model_name, model, train_idx, val_idx, variable)
    for model_name, model in models.items()
    for variable in ["age_group"]
    for train_idx, val_idx in kf.split(X)
)
results_subgroup_m3_agegroup = [x for xs in results_subgroup_m3_agegroup for x in xs]

save_results(
    results_subgroup_m3_agegroup,
    directory,
    n_folds,
    columns=[
        "Model",
        "Variable",
        "Category",
        "AUC",
        "Brier score",
        "Prob true",
        "Prob pred",
        "Train fit time",
        "Test pred time",
    ],
    test_name="SUBGROUP_AGEGROUP",
)

results_subgroup_m3_icu_unit = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(subgroup_analysis)(model_name, model, train_idx, val_idx, variable)
    for model_name, model in models.items()
    for variable in ["ICU_unit"]
    for train_idx, val_idx in kf.split(X)
)
results_subgroup_m3_icu_unit = [x for xs in results_subgroup_m3_icu_unit for x in xs]

save_results(
    results_subgroup_m3_icu_unit,
    directory,
    n_folds,
    columns=[
        "Model",
        "Variable",
        "Category",
        "AUC",
        "Brier score",
        "Prob true",
        "Prob pred",
        "Train fit time",
        "Test pred time",
    ],
    test_name="SUBGROUP_ICU_UNIT",
)


print("SUBGROUP NOISE OVER")

"""### Temporal validation"""

directory = "results/m4/"
create_directory(directory)


def train_evaluate(model_name, model, X_train, y_train, df_test, stratify_on="gender"):


    ## limit training to 10k samples
    X_train  = X_train.sample(N_TRAINING_SAMPLE, random_state=RANDOM_STATE) 
    y_train = y_train.sample(N_TRAINING_SAMPLE, random_state=RANDOM_STATE)

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    train_fit_time = end - start

    start = time.time()
    y_pred = predict_proba_batched(model, df_test[X_train.columns])

    end = time.time()
    test_pred_time = end - start

    stratified_perf = []
    if stratify_on is not None:
        for cat in df_test[stratify_on].dropna().unique():
            X_eval_strat = df_test.loc[df_test[stratify_on] == cat][X_train.columns]
            y_eval_strat = df_test.loc[df_test[stratify_on] == cat][
                "hospital_mortality"
            ]

            start = time.time()
            y_pred = predict_proba_batched(model, X_eval_strat)
            end = time.time()
            test_pred_time = end - start
            prob_true, prob_pred = calibration_curve(y_eval_strat, y_pred)
            stratified_perf.append(
                [
                    model_name,
                    stratify_on,
                    cat,
                    roc_auc_score(y_score=y_pred, y_true=y_eval_strat),
                    brier_score_loss(y_true=y_eval_strat, y_proba=y_pred),
                    prob_true,
                    prob_pred,
                    train_fit_time,
                    test_pred_time,
                ]
            )

        return stratified_perf

    prob_true, prob_pred = calibration_curve(df_test["hospital_mortality"], y_pred)
    return (
        model_name,
        roc_auc_score(y_score=y_pred, y_true=df_test["hospital_mortality"]),
        brier_score_loss(y_true=df_test["hospital_mortality"], y_proba=y_pred),
        prob_true,
        prob_pred,
        train_fit_time,
        test_pred_time,
    )


results_m4_gender = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(train_evaluate)(model_name, model, X, y, mimic_4, stratify_on=variable)
    for model_name, model in models.items()
    for variable in ["gender"]
)

results_m4_gender = [x for xs in results_m4_gender for x in xs]

save_results(
    results_m4_gender,
    directory,
    n_folds,
    columns=[
        "Model",
        "Variable",
        "Category",
        "AUC",
        "Brier score",
        "Prob true",
        "Prob pred",
        "Train fit time",
        "Test pred time",
    ],
    test_name="MIMIC_4_GENDER",
)

results_m4_age = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(train_evaluate)(model_name, model, X, y, mimic_4, stratify_on=variable)
    for model_name, model in models.items()
    for variable in ["age_group"]
)

results_m4_age = [x for xs in results_m4_age for x in xs]

save_results(
    results_m4_age,
    directory,
    n_folds,
    columns=[
        "Model",
        "Variable",
        "Category",
        "AUC",
        "Brier score",
        "Prob true",
        "Prob pred",
        "Train fit time",
        "Test pred time",
    ],
    test_name="MIMIC_4_AGE",
)

results_m4_icu_unit = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(train_evaluate)(model_name, model, X, y, mimic_4, stratify_on=variable)
    for model_name, model in models.items()
    for variable in ["ICU_unit"]
)

results_m4_icu_unit = [x for xs in results_m4_icu_unit for x in xs]

save_results(
    results_m4_icu_unit,
    directory,
    n_folds,
    columns=[
        "Model",
        "Variable",
        "Category",
        "AUC",
        "Brier score",
        "Prob true",
        "Prob pred",
        "Train fit time",
        "Test pred time",
    ],
    test_name="MIMIC_4_ICU_UNIT",
)

results_m4_year = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(train_evaluate)(model_name, model, X, y, mimic_4, stratify_on=variable)
    for model_name, model in models.items()
    for variable in ["anchor_year_group"]
)

results_m4_year = [x for xs in results_m4_year for x in xs]

save_results(
    results_m4_year,
    directory,
    n_folds,
    columns=[
        "Model",
        "Variable",
        "Category",
        "AUC",
        "Brier score",
        "Prob true",
        "Prob pred",
        "Train fit time",
        "Test pred time",
    ],
    test_name="MIMIC_4_YEAR",
)


results_m4 = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(train_evaluate)(model_name, model, X, y, mimic_4, stratify_on=None)
    for model_name, model in models.items()
)

results_m4_df = pd.DataFrame(results_m4)
results_m4_df.to_csv(directory + "m4.csv")


print("SUBGROUP NOISE OVER")
