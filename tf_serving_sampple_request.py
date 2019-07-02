# Imports

import argparse
import json

import numpy as np
import requests

# Arguement parser for passing a test array to evaluate the models
ap = argparse.ArgumentParser()
ap.add_argument("-ta", required=True, help="Pass the array to evaluate the model")
args = vars(ap.parse_args())

test_array = args['ta']

payload = {
    "instances": [{'input_shape':test_array}]
}

# sending post request to TensorFlow Serving server
r = requests.post('http://localhost:9000/v1/models/ImageClassifier:evaluate', json=payload)
model_evaluate_percentage = json.loads(r.content.decode('utf-8'))

#model.evaluate(X_test,Y_test)[1]
