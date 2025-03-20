# york-model-workshop

This repository contains the code for a demo on using production microservice metrics and data to predict system down incidents.

The code can be ran locally, or on IBM WatsonX Notebooks using Python and scikit-learn. Before starting, ensure you have scikit-learn installed:

```
pip install "scikit-learn==1.3.2"
```

Now, execute the next series of commands from the root directory of this repository:
```
cd linalgnorm-0.1
python setup.py sdist --formats=zip
cd ..
mv linalgnorm-0.1/dist/linalgnorm-0.1.zip .
rm -rf linalgnorm-0.1
```

Install the normalizer package:
```
pip install linalgnorm-0.1.zip
```

You can now run the prediction script using different regression models and real data. The `training-dataset.csv` is labelled data from live production systems that we can use to assign the weights to our model. Then, the `test-dataset.csv` is another production use case that has occurred, which we can use to test our generated model to see how early (if at all), it can raise an alert.

```
python prediction_metrics.py
```