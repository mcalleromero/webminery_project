from sklearn.ensemble import RandomForestClassifier
import xgboost as XGBClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier

from skopt import BayesSearchCV

class SyntheticTextClassifier():
    models = {
        'rf': (
            RandomForestClassifier(), 
            {
                'n_estimators': Integer(1, 501),
                'criterion': Categorical(['gini', 'entropy']),
                'max_depth': Integer(-1,15),
                'min_samples_split': Integer(1,10),
                'min_samples_leaf': Integer(1,10),
                'min_weight_fraction_leaf': Real(1e-6, 1e+1, prior='log-uniform'),
                'max_features': Categorical(['auto', 'sqrt', 'log2']),
                'min_impurity_decrease': Real(0, 1e+1, prior='log-uniform'),
            }
        ),
        'xgboost': (
            XGBClassifier(),
            {
                'n_estimators': Integer(1, 501),
                'eta': Real(1e-6, 1e+1, prior='log-uniform'),
                'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
                'max_depth': Integer(-1,15),
                'min_child_weight': Integer(1,15),
                'base_score': Real(0, 1, prior='log-uniform'),
            }
        ),
        'mnb': (
            MultinomialNB(), 
            {
                'alpha': Real(1e-6, 1e+1, prior='log-uniform'),
                'fit_prior': Categorical(['False', 'True']),
            }
        ),
        'gnb': (
            GaussianNB(), 
            {
                'var_smoothing': Real(1e-9, 1e+1, prior='log-uniform'),
            }
        ),
        'bnb': (
            BernoulliNB(), 
            {
                'alpha': Real(1e-6, 1e+1, prior='log-uniform'),
                'fit_prior': Categorical(['False', 'True']),
            }
        ),
        'dt': (
            DecisionTreeClassifier(), 
            {
                'criterion': Categorical(['gini', 'entropy']),
                'max_depth': Integer(-1,15),
                'min_samples_split': Integer(1,10),
                'min_samples_leaf': Integer(1,10),
            }
        ),
    }

    def fit(self, X_train, y_train, model):
        self.clf = BayesSearchCV(models[model][0], models[1])
        return self.clf.fit(X_train, y_train)

    def predict_proba(self, X_test):
        self.clf.predict_proba(X_test)