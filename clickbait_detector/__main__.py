from data_preprocessing import Preprocessing
import joblib

if __name__ == "__main__":
    headline = input("Introduce un headline: ")
    clf_name = input("Introduce el modelo con el que quieres predecir <GradientBoosting, AdaBoost, XGBClassifier, RandomForest, MultinomialNB, GaussianNB, BernoulliNB, DecisionTree: ")

    preprocess = Preprocessing()
    transformed_headline = preprocess.fit_transform(headline)

    clf = joblib.load(f'../models/{clf_name}_model.joblib')
    pred = clf.predict_proba(transformed_headline)

    print(f'El título {headline} tiene un {pred[0,1]*100}% de probabilidades de ser un clickbait.')