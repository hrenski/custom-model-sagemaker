#! /usr/bin/env python3

import argparse
import os
import json
import pandas as pd
import numpy as np
import joblib
from gam_model import GAMRegressor

if __name__ =='__main__':

    print('initializing')
    parser = argparse.ArgumentParser()
    gam = GAMRegressor()
    gam_dict = gam.get_params()


    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type = str, default = os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type = str, default = os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type = str, default = os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--train-file', type = str)
    parser.add_argument('--test', type = str, default = os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--test-file', type = str, default = None)
    
    for argument, default_value in gam_dict.items():
        parser.add_argument(f'--{argument}', type = type(default_value), default = default_value)

        
    print('reading arguments')
    args, _ = parser.parse_known_args()

    print(args)
    
    
    print('setting parameters')
    gam_dict.update({key: value for key, value in vars(args).items() if key in gam_dict and value is not None})
    gam.set_params(**gam_dict)
    
    print(gam)

    
    print('reading training data') 
    # assume there's no headers and the target is the last column
    data = np.loadtxt(os.path.join(args.train, args.train_file), delimiter = ',')
    X = data[:, :-1]
    y = data[:, -1]
    
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if args.test_file is not None:
        print('reading training data') 
        # assume there's no headers and the target is the last column
        data = np.loadtxt(os.path.join(args.test, args.test_file), delimiter = ',')
        X_test = data[:, :-1]
        y_test = data[:, -1]

        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)
    else:
        X_test = None
        y_test = None
    
    
    print('fitting model') 
    gam.fit(X, y)
    
    print("R2 (train):", gam.score(X, y))
    
    if X_test is not None:
        print("R2 (test):", gam.score(X_test, y_test))
    

    print('saving model') 
    path = os.path.join(args.model_dir, "model.joblib")
    print(f"saving to {path}")
    joblib.dump(gam, path)

    
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model
    
def predict_fn(input_object, model):
    return model.predict(input_object)
    
"""

def input_fn(request_body, request_content_type):
    if request_content_type not in ["application/json"]:
        raise RuntimeError("Input request content type ({}) is not supported.".format(request_content_type))
        
    if request_content_type == "application/json":
        data = json.loads(request_body)
        
    return data

def output_fn(prediction, content_type):
    if content_type not in ["application/json"]:
        raise RuntimeError("Response content type ({}) is not supported.".format(request_content_type))
        
    if content_type == "application/json":
        response_body = json.dumps(prediction).encode()
        
    return response_body 
"""    