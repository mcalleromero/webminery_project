from data_preprocessing import Preprocessing
import joblib

if __name__ == "__main__":
    """This main function is used as a simple application that tries to imitate 
    what the performance of the clickbait detector would be in real time executions.
    The flow is simple, the app shows a message that invites you to introduce a news title,
    then, the app asks for a model from the ones previously properly trained and then it just
    returns the headline and the probability it is a clickbait.

    All models must have been previously trained and saved as joblib files in the models directory.
    """
    headline = input("Introduce un headline: ")
    clf_name = input("Introduce el modelo con el que quieres predecir <GradientBoosting, AdaBoost, XGBClassifier, RandomForest, MultinomialNB, GaussianNB, BernoulliNB, DecisionTree>: ")

    preprocess = Preprocessing()
    transformed_headline = preprocess.fit_transform(headline)

    clf = joblib.load(f'../models/{clf_name}_model.joblib')
    pred = clf.predict_proba(transformed_headline)

    print(f'El título {headline} tiene un {round(pred[0,1]*100, 2)}% de probabilidades de ser un clickbait.')