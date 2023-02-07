import os
import logging
import argparse
import numpy as np
import pandas as pd

# import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # MLflow related parameters
    # parser.add_argument("--tracking_uri", type=str)
    # parser.add_argument("--experiment_name", type=str)
    # parser.add_argument("--registered_model_name", type=str)
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--max-leaf-nodes', type=int, default=1)
    parser.add_argument('--max-depth', type=int, default=1)

    # input
    parser.add_argument('--input-dir', type=str, default=os.environ.get('SM_CHANNEL_INPUT'))
    parser.add_argument('--train-file', type=str, default="")
    parser.add_argument('--test-file', type=str, default="")

    args, _ = parser.parse_known_args()

    logging.info('READING DATA')
    # train_df = pd.read_csv(args.train_file)
    # test_df = pd.read_csv(args.test_file)
    
    train_df = pd.read_csv(f'{args.input_dir}/{args.train_file}')
    test_df = pd.read_csv(f'{args.input_dir}/{args.test_file}')

    X_train = train_df.loc[:, train_df.columns != 'target']
    y_train = train_df['target']
    
    X_test = test_df.loc[:, test_df.columns != 'target']
    y_test = test_df['target']

    # set remote mlflow server
    # logging.info('SET EXPERIMENT IN REMOTE MLFLOW SERVER')
    # mlflow.set_tracking_uri(args.tracking_uri)
    # mlflow.set_experiment(args.experiment_name)

    # with mlflow.start_run():
        # params = {
        #     "n-estimators": args.n_estimators,
        #     "min-samples-leaf": args.min_samples_leaf,
        #     "features": args.features
        # }
        # mlflow.log_params(params)

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
    test_f1_score = metrics.f1_score(t_test, t_pred, average='weighted')
    test_metrics = (test_accuracy, test_f1_score)



        # # SAVE MODEL
        # # YOU CAN ADD A METRIC CONDITION HERE BEFORE REGISTERING THE MODEL
        # logging.info('REGISTERING MODEL')
        # # Make sure the IAM role has access to the MLflow bucket
        # mlflow.sklearn.log_model(
        #     sk_model=model,
        #     artifact_path='model',
        #     registered_model_name=args.registered_model_name
        # )
