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
    clf = DummyClassifier(strategy='most_frequent')
    #RANDOM FOREST
    #clf = RandomForestClassifier()
    #NAIVE BAYES
    #clf = MultinomialNB()
    #LOGISTIC REGRESION


    df = data.sample(frac=1).reset_index(drop=True)

    X = df.drop(columns='label')
    y = df['label']

    skf = StratifiedKFold(n_splits=5)

    auc_roc = []

    classifier = SyntheticTextClassifier()

    for clf in tqdm(classifier.models):
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            # Hay que probar que funcione porque no me fio de nuestra capacidad de hacer que las cosas funcionen
            classifier.fit(X_train, y_train, clf)
            preds = classifier.predict_proba(X_test)

            auc_roc.append(metrics.roc_auc_score(y_test, preds[:,1]))

    print(auc_roc)
    print(f'ROC_AUC_mean: {np.mean(auc_roc)}')
