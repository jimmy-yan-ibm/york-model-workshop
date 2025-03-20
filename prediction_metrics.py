import pandas as pd
from linalg_norm.sklearn_transformers import LNormalizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import numpy as np
import warnings
warnings.filterwarnings('ignore')

data_df = pd.read_csv("training-dataset.csv", encoding="utf-8")

print(data_df.head())

target = data_df["status"]
params = data_df.drop("status", axis=1)

lnorm_transf = LNormalizer()
# Create imputer that replaces NaNs with 0
imputer = SimpleImputer(strategy='constant', fill_value=0)
# Choose the Linear Regression model
skl_pipeline = Pipeline(steps=[('normalizer', lnorm_transf), ('imputer', imputer), ('status_estimator', LinearRegression())])
skl_pipeline.fit(params.loc[:, ['container 1 cpu percentage',
                                'container 1 memory percentage',
                                'input container logs',
                                'input journal logs',
                                'input system logs',
                                'output processed logs',
                                'output dropped logs',
                                'output retried logs',
                                'agent restarts',
                                'container 2 cpu percentage',
                                'container 2 memory percentage',
                                'container 3 cpu percentage',
                                'container 3 memory percentage',
                                'input container logs bytes',
                                'input journal logs bytes',
                                'input system logs bytes',
                                'output processed bytes']].values, target)

# The weights (coefficients) indicate the expected change in the status for a one-unit increase in the corresponding parameter.
weights = skl_pipeline.named_steps['status_estimator'].coef_
print("============ Linear Regression Weights ============")
print(weights)

test_df = pd.read_csv("test-dataset.csv", encoding="utf-8")

print(test_df.head())

status_prediction = skl_pipeline.predict(test_df.loc[:, ['container 1 cpu percentage',
                                                        'container 1 memory percentage',
                                                        'input container logs',
                                                        'input journal logs',
                                                        'input system logs',
                                                        'output processed logs',
                                                        'output dropped logs',
                                                        'output retried logs',
                                                        'agent restarts',
                                                        'container 2 cpu percentage',
                                                        'container 2 memory percentage',
                                                        'container 3 cpu percentage',
                                                        'container 3 memory percentage',
                                                        'input container logs bytes',
                                                        'input journal logs bytes',
                                                        'input system logs bytes',
                                                        'output processed bytes']].values)

print(status_prediction)