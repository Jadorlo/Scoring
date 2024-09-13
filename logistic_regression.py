import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve, f1_score, matthews_corrcoef
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

def Create_Train_Test(df):
    """
    Créer les dataframes de test et d'entraînement 
    """
    y = df.pop('income')
    y = pd.get_dummies(y)['>50K']
    X = pd.get_dummies(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    print(X_train.columns)

    return X_train, X_test, y_train, y_test

def Drop_References_Variables(X_train, X_test, vars):
    """
    
    """
    refs_vars = X_train[vars].copy()
    X_train.drop(vars, axis=1, inplace=True)
    X_test.drop(vars, axis=1, inplace=True)

    return refs_vars, X_train, X_test

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

    predict_Y = model.predict(X_test)
    print('Intercept', model.intercept_)
    df_coefficents = pd.DataFrame({'Variables':model.feature_names_in_,
                                   'Coefficients':model.coef_[0],
                                   'Odd-Ratios':np.exp(model.coef_[0])})
    print(df_coefficents.to_string())
    Accuracy = model.score(X_test,y_test)
    class_report = classification_report(y_test, predict_Y, output_dict=True)
    class_report = pd.DataFrame(class_report)
    print(class_report)
    f1_score = class_report.loc['f1-score']['True']
    MCC = matthews_corrcoef(y_test, predict_Y)
    print('Matrice de confusion :\n', confusion_matrix(y_test, predict_Y))
    FER, TER, threshold = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    AUC = auc(FER,TER)

    ### ROC
    plt.figure()
    plt.plot(FER, TER)
    plt.annotate(f'AUC:{round(AUC, 2)}', (0.7,0.3))
    plt.title('Courbe ROC et AUC')
    plt.grid()
    plt.savefig(f'images/ROC_{file}.png')
    # plt.show()

    df_metrics = pd.DataFrame({'Valeur':[AUC, f1_score, Accuracy, MCC],
                               'Cible':[0.8, 0.75, 0.85, 0.75]},
                               index = ['AUC', 'F1-Score', 'Accuracy', 'MCC'])
    return df_metrics

def Scoring(df_metrics):
    """
    
    """
    print("#### SCORING ####")
    print(df_metrics)
    print('AUC:', df_metrics.loc['AUC']['Valeur'])
    print('f1_score:', df_metrics.loc['F1-Score']['Valeur'])
    print('Accuracy:', df_metrics.loc['Accuracy']['Valeur'])
    print('MCC:', df_metrics.loc['MCC']['Valeur'])

    score = df_metrics.apply(lambda row : 1 if row['Valeur'] > row['Cible'] else 0, axis=1).sum()
    print('Score du Logit:', score)

def main():
    df = pd.read_csv(args.filename)
    X_train, X_test, y_train, y_test = Create_Train_Test(df)
    vars = ['native-country_United-States', 'workclass_Private', 'occupation_Prof-specialty', 
            'gender_Male', 'education_HS-grad', 'relationship_Husband', 'marital-status_Married-civ-spouse',
            'race_White']
    refs_vars, X_train, X_test = Drop_References_Variables(X_train, X_test, vars)
    model = Logistic_Regression(X_train, y_train)
    df_metrics = Evaluation(model, X_test, y_test)
    Scoring(df_metrics)
    



if __name__ == "__main__":
    main()