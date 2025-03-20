import pandas as pd
from linalg_norm.sklearn_transformers import LNormalizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
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
skl_pipeline_1 = Pipeline(steps=[('normalizer', lnorm_transf), ('imputer', imputer), ('status_estimator', LinearRegression())])
skl_pipeline_1.fit(params.loc[:, ['container 1 cpu percentage',
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
weights_1 = skl_pipeline_1.named_steps['status_estimator'].coef_
print("============ Linear Regression Weights ============")
print(weights_1)

test_df = pd.read_csv("test-dataset.csv", encoding="utf-8")

print(test_df.head())

status_prediction_1 = skl_pipeline_1.predict(test_df.loc[:, ['container 1 cpu percentage',
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

print("============ Linear Regression Prediction ============")
prediction_df_1 = test_df.copy()
prediction_df_1['status_prediction'] = status_prediction_1

plt.figure(figsize=(10, 6))
plt.plot(prediction_df_1['Time'], prediction_df_1['status_prediction'])
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
plt.xlabel('Timestamp')
plt.ylabel('Predicted Status')
plt.title('Predicted Status Over Time (Linear Regression)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Choose the Ride model, linear regression with penalty for large coefficients
skl_pipeline_2 = Pipeline(steps=[('normalizer', lnorm_transf), ('imputer', imputer), ('status_estimator', Ridge(alpha=1.0))])
skl_pipeline_2.fit(params.loc[:, ['container 1 cpu percentage',
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

weights_2 = skl_pipeline_2.named_steps['status_estimator'].coef_
print("============ Ridge Weights ============")
print(weights_2)

status_prediction_2 = skl_pipeline_2.predict(test_df.loc[:, ['container 1 cpu percentage',
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

print("============ Ridge Prediction ============")
prediction_df_2 = test_df.copy()
prediction_df_2['status_prediction'] = status_prediction_2

plt.figure(figsize=(10, 6))
plt.plot(prediction_df_2['Time'], prediction_df_2['status_prediction'])
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
plt.xlabel('Timestamp')
plt.ylabel('Predicted Status')
plt.title('Predicted Status Over Time (Ridge)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




# Choose the Random Forest Regression model --> 500 decision trees, randomized subsets of data
skl_pipeline_3 = Pipeline(steps=[('normalizer', lnorm_transf), ('imputer', imputer), ('status_estimator', RandomForestRegressor(n_estimators=500, random_state=42))])
skl_pipeline_3.fit(params.loc[:, ['container 1 cpu percentage',
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

importances = skl_pipeline_3.named_steps['status_estimator'].feature_importances_
print("============ FEATURE IMPORTANCE (Random Forest) ===========")
print(importances)

status_prediction_3 = skl_pipeline_3.predict(test_df.loc[:, ['container 1 cpu percentage',
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

print("============ Random Forest Regression Prediction ============")
prediction_df_3 = test_df.copy()
prediction_df_3['status_prediction'] = status_prediction_3

plt.figure(figsize=(10, 6))
plt.plot(prediction_df_3['Time'], prediction_df_3['status_prediction'])
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
plt.xlabel('Timestamp')
plt.ylabel('Predicted Status')
plt.title('Predicted Status Over Time (Random Forest Regression)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()