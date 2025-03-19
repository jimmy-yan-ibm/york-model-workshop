# york-model-workshop

This repository contains the skeleton code for a workshop on using production microservice metrics and data to predict system down incidents.

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

You can now run the prediction script using a simple Linear Regression model and dummy data. Feel free to play around - we will be using more exhaustive datasets in the workshop.

```
python prediction_metrics.py
```