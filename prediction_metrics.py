import pandas as pd
from linalg_norm.sklearn_transformers import LNormalizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

dummy_data = [
    [0, 10, 100, 4000, 0, 0, 0, 11, 0, 0, 0, 0],
    [50, 3, 1000, 1000, 3, 3000, 55, 0, 2, 7, 10000, 1],
    [0, 80, 10000, 2000, 45, 10, 10, 0, 2, 0, 0, 0],
]
columns = [ 'pod restarts',
            'vault memory percentage',
            'input container logs',
            'input journal logs',
            'memory percentage',
            'input sys logs',
            'cpu percentage',
            'output ICL processed logs',
            'vault cpu percentage',
            'output ICL dropped logs',
            'output ICL retried logs',
            'status'
        ]

data_df = pd.DataFrame(dummy_data, columns=columns)
target = data_df["status"]
params = data_df.drop("status", axis=1)

# # Could split into train/test sets here
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(params, target, test_size=0.25, random_state=143)

lnorm_transf = LNormalizer()
# Create imputer that replaces NaNs with 0
imputer = SimpleImputer(strategy='constant', fill_value=0)
# Choose the Linear Regression model
skl_pipeline = Pipeline(steps=[('normalizer', lnorm_transf), ('imputer', imputer), ('status_estimator', LinearRegression())])
skl_pipeline.fit(params.loc[:, ['pod restarts',
                                'vault memory percentage',
                                'input container logs',
                                'input journal logs',
                                'memory percentage',
                                'input sys logs',
                                'cpu percentage',
                                'output ICL processed logs',
                                'vault cpu percentage',
                                'output ICL dropped logs',
                                'output ICL retried logs']].values, target)

# The weights (coefficients) indicate the expected change in the status for a one-unit increase in the corresponding parameter.
weights = skl_pipeline.named_steps['status_estimator'].coef_
print(weights)

test_data = [
    [20, 10, 3000, 4000, 5, 0, 43, 0, 21, 500, 0]
]
columns = [ 'pod restarts',
            'vault memory percentage',
            'input container logs',
            'input journal logs',
            'memory percentage',
            'input sys logs',
            'cpu percentage',
            'output ICL processed logs',
            'vault cpu percentage',
            'output ICL dropped logs',
            'output ICL retried logs'
        ]

test_df = pd.DataFrame(test_data, columns=columns)

status_prediction = skl_pipeline.predict(test_df.loc[:, ['pod restarts',
                                                        'vault memory percentage',
                                                        'input container logs',
                                                        'input journal logs',
                                                        'memory percentage',
                                                        'input sys logs',
                                                        'cpu percentage',
                                                        'output ICL processed logs',
                                                        'vault cpu percentage',
                                                        'output ICL dropped logs',
                                                        'output ICL retried logs']].values)

print(status_prediction)