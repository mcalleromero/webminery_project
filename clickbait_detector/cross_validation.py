import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
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

import joblib

DATA = Path('../data')

if __name__ == "__main__":
    """This function's purpose is to train each model with their optimized parameters
    and save them as joblib binary files in the models directory, in order to use them
    in the future. This method also saves a file within the metrics evaluated are presented.
    """
    today = date.today().strftime("%d-%m-%Y")
    filename = f'title_dataset_{today}.csv'
    final_file_path = DATA / filename
    data = pd.read_csv(final_file_path)
    
    models = {
        'GradientBoosting': GradientBoostingClassifier(criterion='friedman_mse', init=None,
                                learning_rate=0.1, loss='deviance', max_depth=3,
                                max_features=None, max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=100,
                                n_iter_no_change=None, presort='auto',
                                random_state=None, subsample=1.0, tol=0.0001,
                                validation_fraction=0.1, verbose=0,
                                warm_start=False),
        'AdaBoost': AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                        n_estimators=50, random_state=None),
        'XGBClassifier': XGBClassifier(base_score=None, booster=None, colsample_bylevel=None,
                    colsample_bynode=None, colsample_bytree=None, gamma=None,
                    gpu_id=None, importance_type='gain', interaction_constraints=None,
                    learning_rate=None, max_delta_step=None, max_depth=None,
                    min_child_weight=None, monotone_constraints=None,
                    n_estimators=100, n_jobs=None, num_parallel_tree=None,
                    objective='binary:logistic', random_state=None, reg_alpha=None,
                    reg_lambda=None, scale_pos_weight=None, subsample=None,
                    tree_method=None, use_label_encoder=True,
                    validate_parameters=None, verbosity=None),
        'RandomForest': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                            max_depth=None, max_features='auto', max_leaf_nodes=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, n_estimators='warn',
                            n_jobs=None, oob_score=False, random_state=None,
                            verbose=0, warm_start=False),
        'MultinomialNB': MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True),
        'GaussianNB': GaussianNB(priors=None, var_smoothing=1e-09),
        'BernoulliNB': BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True),
        'DecisionTree': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                            max_features=None, max_leaf_nodes=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, presort=False,
                            random_state=None, splitter='best')
    }


    df = data.sample(frac=1).reset_index(drop=True)

    X = df.drop(columns='label')
    y = df['label']

    metricsFile = open("../data/metrics_kfolds.csv", "w+")
    metricsFile.write(f'name,auc,accuracy,precision,f1_score,recall,log_loss\n')

    skf = StratifiedKFold(n_splits=5)

    for clf_name in tqdm(models):
        auc = []
        accuracy = []
        precision = []
        f1_score = []
        recall = []
        log_loss = []

        clf = models[clf_name]

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            clf.fit(X_train, y_train)
            preds_proba = clf.predict_proba(X_test)
            preds = clf.predict(X_test)

            auc.append(metrics.roc_auc_score(y_test, preds_proba[:,1]))
            accuracy.append(metrics.accuracy_score(y_test, preds))
            precision.append(metrics.precision_score(y_test, preds))
            f1_score.append(metrics.f1_score(y_test, preds))
            recall.append(metrics.recall_score(y_test, preds))
            log_loss.append(metrics.log_loss(y_test, preds_proba[:,1]))

        metricsFile.write(f'{clf_name},{round(np.mean(auc),3)}+-{round(np.std(auc),3)},{round(np.mean(accuracy),3)}+-{round(np.std(accuracy),3)},{round(np.mean(precision),3)}+-{round(np.std(precision),3)},{round(np.mean(f1_score),3)}+-{round(np.std(f1_score), 3)},{round(np.mean(recall),3)}+-{round(np.std(recall),3)},{round(np.mean(log_loss),3)}+-{round(np.std(log_loss),3)}\n')
    
        clf.fit(X, y)
        models_file = f'../models/{clf_name}_model.joblib'
        joblib.dump(clf, models_file)

    metricsFile.close()