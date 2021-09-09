#%% [markdown]
## Monotonicity
# A non increasing or non decreasing function is said to be monotonic i.e. when $\mathbf{x}<\mathbf{y}$ and $\mathbf{f(x)}<\mathbf{f(y)}$ (monotonically increasing) or when $\mathbf{x}<\mathbf{y}$ and $\mathbf{f(x)}>\mathbf{f(y)}$ (monotonically decreasing)

# ![Monotonicity](monotonicity.png)

#%%
import warnings
from typing import List

import catboost as cb
import dalex as dx
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.experimental import (  # this is still experimental
    enable_hist_gradient_boosting,
)

# %%
warnings.filterwarnings("ignore", category=FutureWarning)
# back to the default behavior
# warnings.filterwarnings("default", category=FutureWarning)

# %%
def target_df(x1: List, x2: List):
    """creates target using mathematically generated data
    (xgboost documentation)
    """
    return [
        (5 * x1)
        + np.sin(10 * np.pi * x1)
        - (5 * x2)
        - np.cos(10 * np.pi * x2)
        + np.random.normal(0, 0.01)
        for x1, x2 in zip(x1, x2)
    ]


def check_trend(model_profile: dx.Explainer.model_profile):
    """extract pdp values from dalex explainer object"""
    pdps = pd.DataFrame(model_profile.result)
    return [pdps[pdps["_vname_"] == i] for i in pdps["_vname_"].unique()]


def plot_pdps(pdps: List, dfs_list: List, target: List):
    """utility to plot pdps extracted from dalex package"""
    rows = len(pdps)
    fig, axes = plt.subplots(nrows=rows, figsize=(12, 12))
    for p, d, ax in zip(pdps, dfs_list, axes):
        ax.scatter(d, target, facecolor="none", edgecolors="blue", alpha=0.4)
        ax.plot(p["_x_"], p["_yhat_"], color="k")


# %%
# Generating random 500 points between 0 and 1
x1 = [np.random.random() for i in range(500)]
x2 = [np.random.random() for i in range(500)]
target = pd.Series(target_df(x1, x2))

# %%

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 12))
ax1.scatter(x1, target, facecolor="none", edgecolors="blue")
ax1.set_xlabel("x1")

ax2.scatter(x2, target, facecolor="none", edgecolors="red")
ax2.set_xlabel("x2")

# %% [markdown]
## XGBoost

#%%

pd_train = pd.DataFrame(pd.concat([pd.Series(x1), pd.Series(x2)], axis=1))
dtrain = xgb.DMatrix(pd_train, label=target)

params = {
    "max_depth": 2,
    "eta": 1,
    "objective": "reg:squarederror",
    "nthread": 4,
    "eval_metric": "rmse",
}

#%%

xgmod_noconst = xgb.train(params, dtrain, num_boost_round=1000)
exp_xg_noconst = dx.Explainer(xgmod_noconst, pd_train, target)


# %%

exp_xg_noconst_p = exp_xg_noconst.model_profile()
pdps_xg_noconst = check_trend(exp_xg_noconst_p)
plot_pdps(pdps_xg_noconst, [x1, x2], target)

#%%
xgparams_constrained = params.copy()
xgparams_constrained["monotone_constraints"] = "(1,-1)"

# %%

xgmod_with_const = xgb.train(xgparams_constrained, dtrain, num_boost_round=1000)

#%%
exp_xg_const = dx.Explainer(xgmod_with_const, pd_train, target)

exp_xg_const_p = exp_xg_const.model_profile()
pdps_xg_const = check_trend(exp_xg_const_p)
plot_pdps(pdps_xg_const, [x1, x2], target)

# %% [markdown]

## Light GBM

# %%

lgb_train = lgb.Dataset(pd_train, target)

# %%
lgb_params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "verbose": -1,
}

lgbmod_noconst = lgb.train(lgb_params, lgb_train, num_boost_round=1000)
exp_lgb_noconst = dx.Explainer(lgbmod_noconst, pd_train, target)

exp_lgb_noconst_p = exp_lgb_noconst.model_profile()
pdps_lgb_noconst = check_trend(exp_lgb_noconst_p)
plot_pdps(pdps_lgb_noconst, [x1, x2], target)

# %%

lgb_params_constrained = lgb_params.copy()
lgb_params_constrained["monotone_constraints"] = [1, -1]

lgbmod_const = lgb.train(lgb_params_constrained, lgb_train, num_boost_round=1000)

#%%

exp_lgb_const = dx.Explainer(lgbmod_const, pd_train, target)

exp_lgb_const_p = exp_lgb_const.model_profile()
pdps_lgb_const = check_trend(exp_lgb_const_p)
plot_pdps(pdps_lgb_const, [x1, x2], target)

# %% [markdown]

# There are various other models/packages which allow us to enforce monotonicity, for e.g.
# * Splines [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html), [stackoverflow answer based on R](https://stats.stackexchange.com/questions/197509/how-to-smooth-data-and-force-monotonicity)
# * Hist Gradient Boosting [scikit](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
# * Catboost [(search for **monotone_constraints** upon clicking this)](https://catboost.ai/docs/concepts/python-reference_parameters-list.html)
# * TensorFlow Lattice [(TFL)](https://www.tensorflow.org/lattice/overview)

# %%

# %%
