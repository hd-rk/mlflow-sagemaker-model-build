import os
import logging
import argparse
import numpy as np
import pandas as pd

import mlflow.sklearn
import mlflow

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # MLflow related parameters
    parser.add_argument("--mlflow-tracking-uri", type=str)
    parser.add_argument("--mlflow-experiment-name", type=str)
    parser.add_argument("--mlflow-model-name", type=str)
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--max-leaf-nodes', type=int, default=1)
    parser.add_argument('--max-depth', type=int, default=1)

    # input
    parser.add_argument('--input-dir', type=str, default=os.environ.get('SM_CHANNEL_INPUT'))
    parser.add_argument('--train-file', type=str, default="")
    parser.add_argument('--test-file', type=str, default="")

    args, _ = parser.parse_known_args()

    logging.info('READING DATA')
    
    train_df = pd.read_csv(f'{args.input_dir}/{args.train_file}')
    test_df = pd.read_csv(f'{args.input_dir}/{args.test_file}')

    X_train = train_df.loc[:, train_df.columns != 'target']
    y_train = train_df['target']
    
    X_test = test_df.loc[:, test_df.columns != 'target']
    y_test = test_df['target']

    # set remote mlflow server
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run():
        params = {
            "max-leaf-nodes": args.max_leaf_nodes,
            "max-depth": args.max_depth,
        }
        mlflow.log_params(params)

        # TRAIN
        logging.info('TRAINING MODEL')
        classifier = DecisionTreeClassifier(
                random_state=42, 
                max_leaf_nodes=args.max_leaf_nodes, 
                max_depth=args.max_depth,
        )        
        classifier.fit(X_train, y_train)

        logging.info('EVALUATING MODEL')
        y_pred = classifier.predict(X_test)
        test_accuracy = metrics.accuracy_score(y_test, y_pred)
        test_f1_score = metrics.f1_score(y_test, y_pred, average='weighted')
        logging.info(f'metric_accuracy={test_accuracy}')
        logging.info(f'metric_f1={test_f1_score}')
        
        mlflow.log_metric('accuracy', test_accuracy)
        mlflow.log_metric('f1', test_f1_score)

        # # SAVE MODEL
        # # YOU CAN ADD A METRIC CONDITION HERE BEFORE REGISTERING THE MODEL
        # logging.info('REGISTERING MODEL')
        # # Make sure the IAM role has access to the MLflow bucket
        # mlflow.sklearn.log_model(
        #     sk_model=model,
        #     artifact_path='model',
        #     registered_model_name=args.registered_model_name
        # )
