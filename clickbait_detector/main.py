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
    today = date.today().strftime("%d-%m-%Y")
    filename = f'title_dataset_{today}.csv'
    final_file_path = DATA / filename
    data = pd.read_csv(final_file_path)
    
    #MODELS    
    #DUMMY CLASSIFIER
    #clf = DummyClassifier(strategy='most_frequent')
    #RANDOM FOREST
    #clf = RandomForestClassifier()
    #NAIVE BAYES
    #clf = MultinomialNB()
    #LOGISTIC REGRESION


    df = data.sample(frac=1).reset_index(drop=True)

    X = df.drop(columns='label')
    y = df['label']

    X_train = X.head(int(len(df)*0.66))
    y_train = y.head(int(len(df)*0.66))
    X_test = X.tail(int(len(df)*0.33))
    y_test = y.tail(int(len(df)*0.33))

    #skf = StratifiedKFold(n_splits=5)

    auc_roc = []

    classifier = SyntheticTextClassifier()

    params = open("../data/params.txt", "w+")
    params.write(f'name,params,auc,accuracy,precision,f1_score,recall,log_loss')

    for clf_name in tqdm(classifier.models):
        clf = classifier.fit(X_train, y_train, clf_name)
        preds_proba = clf.predict_proba(X_test)
        preds = clf.predict(X_test)

        # for train_index, test_index in skf.split(X, y):
        #     X_train, X_test = X.loc[train_index], X.loc[test_index]
        #     y_train, y_test = y.loc[train_index], y.loc[test_index]

        #     # Hay que probar que funcione porque no me fio de nuestra capacidad de hacer que las cosas funcionen
        #     classifier.fit(X_train, y_train, clf)
        #     preds = classifier.predict_proba(X_test)
        auc = metrics.roc_auc_score(y_test, preds_proba[:,1])
        accuracy = metrics.accuracy_score(y_test, preds)
        precision = metrics.precision_score(y_test, preds)
        f1_score = metrics.f1_score(y_test, preds)
        recall = metrics.recall_score(y_test, preds)
        log_loss = metrics.log_loss(y_test, preds_proba[:,1])

        print(auc)
        params.write(f'{clf_name},{clf.get_params()},{auc},{accuracy},{precision},{f1_score},{recall},{log_loss}\n')
    
    params.close()

    # print(auc_roc)
    # print(f'ROC_AUC_mean: {np.mean(auc_roc)}')
