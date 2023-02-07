import argparse
import logging
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    logger.debug("Running data preparation...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="")
    args, _ = parser.parse_known_args()

    # Load the Iris dataset from sk-learn
    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=42)

    trainX = pd.DataFrame(X_train, columns=data.feature_names)
    trainX['target'] = y_train

    testX = pd.DataFrame(X_test, columns=data.feature_names)
    testX['target'] = y_test

    # save train and test CSV files
    trainX.to_csv(f'{args.output}/iris_train.csv')
    testX.to_csv(f'{args.output}/iris_test.csv')
