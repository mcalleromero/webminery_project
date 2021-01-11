
import pandas as pd
from datetime import date
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

DATA = Path('../data')

if __name__ == "__main__":
    today = date.today().strftime("%d-%m-%Y")
    filename = f'title_dataset_{today}.csv'
    final_file_path = DATA / filename
    data = pd.read_csv(final_file_path)
    

    df = data.sample(frac=1).reset_index(drop=True)
    
    df_train = df.head(int(len(df) * 0.66))
    df_test = df.tail(int(len(df) * 0.33))
    
    X_train = df_train.drop(columns='label')
    y_train = df_train['label']

    X_test = df_test.drop(columns='label')
    y_test = df_test['label']

    #clf = RandomForestClassifier()
    clf = MultinomialNB()

    clf.fit(X_train, y_train)
    
    preds = clf.predict_proba(X_test)

    print(f'ROC_AUC: {metrics.roc_auc_score(y_test, preds[:,1])}')
