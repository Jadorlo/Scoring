import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve, matthews_corrcoef
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import argparse

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0, stratify=y)

    return X_train, X_test, y_train, y_test

def foret_aleatoire(X_train, y_train):
    """
    Compute Random Forest algo
    """
    model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=10).fit(X_train, y_train)
    return model

def Evaluation(model, X_test, y_test):

    """
    Evaluation
    """

    predict_Y = model.predict(X_test)
    names = model.feature_names_in_

    df_importance = pd.DataFrame({'Variables':names,
                                      'Importance':model.feature_importances_}).sort_values(by='Importance',
                                                                                            ascending=False, ignore_index=True)
    print(df_importance.loc[df_importance['Importance']!=0])

    print('Matrice de confusion:\n', confusion_matrix(y_test, predict_Y))
    Accuracy = model.score(X_test,y_test)
    class_report = classification_report(y_test, predict_Y, output_dict=True)
    class_report = pd.DataFrame(class_report)
    f1_score = class_report.loc['f1-score']['True']
    MCC = matthews_corrcoef(y_test, predict_Y)
    AUC = ROC(model, X_test, y_test)

    df_metrics = pd.DataFrame({'Valeur':[AUC, f1_score, Accuracy, MCC, None],
                               'Cible':[0.8, 0.75, 0.85, 0.75, None]},
                               index = ['AUC', 'F1-Score', 'Accuracy', 'MCC', 'gAUC'])
    return df_metrics




def ROC(model, X_test, y_test):
    """
    Charge la courbe roc du modèle et retourne l'AUC
    """
    FER, TER, threshold = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    AUC = auc(FER,TER)

    ### ROC ####
    plt.figure()
    plt.plot(FER, TER)
    plt.annotate(f'AUC:{round(AUC, 2)}', (0.7,0.3))
    plt.title('Courbe ROC et AUC')
    plt.grid()
    #plt.savefig(f'images/ROC/ROC_{file}.png')
    plt.show()

    return AUC

def Scoring(df_metrics):
    """
    Calcul le score du modèle en fonction des métriques de df_metrics
    """

    print(f"#### SCORING####")
    print(df_metrics)
    score = df_metrics.apply(lambda row : 1 if row['Valeur'] > row['Cible'] else 0, axis=1).sum()
    print(f'Score :', score)
    return df_metrics, score

def main(df):
    """
    
    """
    X_train, X_test, y_train, y_test = Create_Train_Test(df)
    model_decision_tree = foret_aleatoire(X_train, y_train)
    df_metrics_tree = Evaluation(model_decision_tree, X_test, y_test)
    df_metrics_tree, score = Scoring(df_metrics_tree)
    #df_tracking = Tracking_Dataframe(model_decision_tree.get_params(), df_metrics_tree, score, False)

df = pd.read_csv(args.filename)
main(df)