#%% [markdown]

## Linear Models in Scikit vs Statsmodels
# **One must know the difference**

# %%

import lightgbm as lgb
import numpy as np

# Importing modules
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_boston, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression

#%%

X, y = load_boston(return_X_y=True)
feature_names = load_boston().feature_names
X_pd = pd.DataFrame(X, columns=feature_names)

# %% [markdown]

# **Here is a list of features that the data has along with their description**
# * VCRIM per capita crime rate by town
# * ZN proportion of residential land zoned for lots over 25,000 sq.ft.
# * INDUS proportion of non-retail business acres per town
# * CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# * NOX nitric oxides concentration (parts per 10 million)
# * RM average number of rooms per dwelling
# * AGE proportion of owner-occupied units built prior to 1940
# * DIS weighted distances to five Boston employment centres
# * RAD index of accessibility to radial highways
# * TAX full-value property-tax rate per $10,000
# * PTRATIO pupil-teacher ratio by town
# * B 1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
# * LSTAT % lower status of the population
# * MEDV Median value of owner-occupied homes in $1000’s

# %%

# statsmodels lr model (data at original scale)
X_pd_sm = sm.add_constant(X_pd)
sm_lr = sm.OLS(y, X_pd_sm)
sm_lr_res = sm_lr.fit()
sm_summary1 = sm_lr_res.summary()
# %%
# scikit-learn lr model (data at original scale)
sk_lr = LinearRegression()
sk_lr.fit(X_pd, y)

# creating a pandas dataframe of coefficients
sk_coefs = pd.DataFrame(sk_lr.coef_, feature_names, columns=["coeficients"])

#%%

print("\n", "#" * 20, "Statsmodels Summary default output", "#" * 20)
print(sm_summary1)

print("\n", "#" * 20, "Scikit OLS coefficients output", "#" * 20)
print(sk_coefs)

#%%

(sk_coefs.values.ravel() * np.std(X_pd)).sort_values()
#%%
np.std(X_pd)


# %%
from sklearn.preprocessing import StandardScaler

# %%
std = StandardScaler()
X_std = std.fit_transform(X_pd)
X_std_pd = pd.DataFrame(X_std, columns=feature_names)
# %%
# statsmodels lr model (data at original scale)
X_std_pd_sm = sm.add_constant(X_std_pd)
sm_lr_std = sm.OLS(y, X_std_pd_sm)
sm_lr_std_res = sm_lr_std.fit()
sm_summary2 = sm_lr_std_res.summary()
# %%
# scikit-learn lr model (data at original scale)
sk_std_lr = LinearRegression()
sk_std_lr.fit(X_std_pd, y)

# creating a pandas dataframe of coefficients
sk_std_coefs = pd.DataFrame(sk_std_lr.coef_, feature_names, columns=["coeficients"])

#%%
print("\n", "#" * 20, "Statsmodels Summary default output", "#" * 20)
print(sm_summary2)
print("\n", "#" * 20, "Scikit OLS coefficients output", "#" * 20)
print(sk_std_coefs)
#%%
sk_std_coefs.sort_values(by="coeficients")


# %%
pd1 = pd.DataFrame(sm_summary2.tables[1])
# pd1.columns = pd1[0]
# pd1.drop(0,axis = 0,inplace=True)
# %%
sm_coefs = pd1[1]
sm_t_val = pd1[2]
# %%
sm_coefs.drop(0, axis=0, inplace=True)

# /sm_t_val
# %%

lgb_data = lgb.Dataset(X_pd, label=y)

# %%
params = {"learning_rate": 0.01, "verbose": -1}
lgb_model = lgb.train(params=params, train_set=lgb_data)

# %%
lgb.plot_importance(lgb_model)

#%%
