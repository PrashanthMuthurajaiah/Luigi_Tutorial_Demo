# Imports

import argparse
import json

import numpy as np
import requests

# Arguement parser for passing a test array to evaluate the models
ap = argparse.ArgumentParser()
ap.add_argument("--xtest",required=True, help="Pass the X_test array for evaluation")
ap.add_argument("--ytest",required=True, help="Pass the Y_test array for evaluation")

args = vars(ap.parse_args())

X_test = args['xtest']
Y_test = args['ytest']

X_test_np = np.load(X_test)
Y_test_np = np.load(Y_test)

X_test_np_list = X_test_np.to_list()
Y_test_np_list = Y_test_np.to_list()

payload = {
    "instances": [{'X_test':X_test_np_list,'Y_test':Y_test_np_list}]
}

# sending post request to TensorFlow Serving server
r = requests.post('http://localhost:9000/v1/models/price_predictor:evaluate', json=payload)
model_evaluate_percentage = json.loads(r.content.decode('utf-8'))
print(model_evaluate_percentage)

#model.evaluate(X_test,Y_test)[1]
