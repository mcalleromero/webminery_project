import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV

class SyntheticTextClassifier:
    """This class is prepared to test several ML classifiers and its parameters
    with the aim of extracting their optimized ones.

    Returns:
        [SyntheticTextClassifier]: An instance of the class to test the models and its parameters
    """
    models = {
        'gradientboosting': (
            GradientBoostingClassifier(), 
            {
                'n_estimators': Integer(1, 501),
                'loss': Categorical(['deviance', 'exponential']),
                'learning_rate': Real(1e-6, 1e+1, prior='log-uniform'),
                'criterion': Categorical(['friedman_mse', 'mse', 'mae']),
                'min_samples_leaf': Integer(1,10),
                'min_weight_fraction_leaf': Real(1e-6, 0.5, prior='log-uniform'),
                'max_depth': Integer(3,10),
                'max_features': Categorical(['auto', 'sqrt', 'log2']),
                'min_impurity_decrease': Real(1e-6, 1e+1, prior='log-uniform'),
            }
        ),
        'adaboost': (
            AdaBoostClassifier(), 
            {
                'n_estimators': Integer(1, 501),
                'learning_rate': Real(1e-6, 1e+1, prior='log-uniform'),
                'algorithm': Categorical(['SAMME', 'SAMME.R']),
            }
        ),
        'xgboost': (
            XGBClassifier(),
            {
                'n_estimators': Integer(1, 501),
                'eta': Real(1e-6, 1e+1, prior='log-uniform'),
                'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
                'min_child_weight': Integer(1,15),
            }
        ),
        'rf': (
            RandomForestClassifier(), 
            {
                'n_estimators': Integer(1, 501),
                'criterion': Categorical(['gini', 'entropy']),
                'min_samples_split': Integer(2,10),
                'min_samples_leaf': Integer(1,10),
                'max_features': Categorical(['auto', 'sqrt', 'log2']),
                'min_impurity_decrease': Real(1e-6, 1e+1, prior='log-uniform'),
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
                'max_depth': Integer(1,15),
                'min_samples_split': Integer(2,10),
                'min_samples_leaf': Integer(1,10),
            }
        ),
        
    }

    def fit(self, X_train, y_train, model):
        self.clf = BayesSearchCV(self.models[model][0], self.models[model][1], n_iter=32, iid=False)
        return self.clf.fit(X_train, y_train)

    def predict_proba(self, X_test):
        self.clf.predict_proba(X_test)