import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

def create_train_test(df):
    """
    Créer les dataframes de test et d'entraînement 
    """
    y = df.pop('income')
    y = pd.get_dummies(y)['>50K']
    X = pd.get_dummies(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    return X_train, X_test, y_train, y_test

def Logistic_Regression(X_train, y_train):
    """
    Créer le modèle grâce aux datasets d'entraînement
    """
    model = LogisticRegression(random_state = 0).fit(X_train, y_train)

    return model

def Evaluation(model, X_test, y_test):
    """
    Affiche les résultats du modèle avec les datasets de test
    """
    file = args.filename.split('.')[0].split('/')[-1]

    predict_X = model.predict(X_test)
    print('Intercept', model.intercept_)
    df_coefficents = pd.DataFrame({'Variables':model.feature_names_in_,
                                   'Coefficients':model.coef_[0],
                                   'Odd-Ratios':np.exp(model.coef_[0])})
    print(df_coefficents.to_string())
    print('Accuracy', model.score(X_test,y_test))
    print('Matrice de confusion\n', confusion_matrix(y_test, predict_X))
    print(classification_report(y_test, predict_X))
    FER, TER, threshold = roc_curve(y_test, model.predict_proba(X_test)[:,1])

    ### ROC
    plt.figure()
    plt.plot(FER, TER)
    plt.annotate(f'AUC:{round(auc(FER,TER), 2)}', (0.7,0.3))
    plt.savefig(f'images/ROC_{file}.png')
    # plt.show()


def main():
    df = pd.read_csv(args.filename)
    X_train, X_test, y_train, y_test = create_train_test(df)
    model = Logistic_Regression(X_train, y_train)
    Evaluation(model, X_test, y_test)



if __name__ == "__main__":
    main()