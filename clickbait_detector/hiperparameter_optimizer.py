import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import date
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier

from models import SyntheticTextClassifier

DATA = Path('../data')

if __name__ == "__main__":
    """This function purpose is to extract the optimized parameters of each model and
    compare their performances with evaluation metrics such as f1 score, accuracy or
    the auc from their roc curve.
    """
    today = date.today().strftime("%d-%m-%Y")
    filename = f'title_dataset_{today}.csv'
    final_file_path = DATA / filename
    data = pd.read_csv(final_file_path)

    df = data.sample(frac=1).reset_index(drop=True)

    X = df.drop(columns='label')
    y = df['label']

    X_train = X.head(int(len(df)*0.66))
    y_train = y.head(int(len(df)*0.66))
    X_test = X.tail(int(len(df)*0.33))
    y_test = y.tail(int(len(df)*0.33))

    auc_roc = []

    classifier = SyntheticTextClassifier()

    paramsFile = open("../data/params2.csv", "w+")
    metricsFile = open("../data/metrics2.csv", "w+")
    paramsFile.write(f'name,params\n')
    metricsFile.write(f'name,auc,accuracy,precision,f1_score,recall,log_loss\n')

    for clf_name in tqdm(classifier.models):
        clf = classifier.fit(X_train, y_train, clf_name)
        preds_proba = clf.predict_proba(X_test)
        preds = clf.predict(X_test)

        auc = metrics.roc_auc_score(y_test, preds_proba[:,1])
        accuracy = metrics.accuracy_score(y_test, preds)
        precision = metrics.precision_score(y_test, preds)
        f1_score = metrics.f1_score(y_test, preds)
        recall = metrics.recall_score(y_test, preds)
        log_loss = metrics.log_loss(y_test, preds_proba[:,1])

        paramsFile.write(f'{clf_name},{clf.get_params()["estimator"]}\n')
        metricsFile.write(f'{clf_name},{auc},{accuracy},{precision},{f1_score},{recall},{log_loss}\n')
    
    paramsFile.close()
    metricsFile.close()
