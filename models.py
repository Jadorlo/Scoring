import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve, matthews_corrcoef
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
    Crée le Logit grâce aux datasets d'entraînement
    """
    model = LogisticRegression(random_state = 0).fit(X_train, y_train)

    return model

def DecisionTree(X_train, y_train):
    """
    Créer l'arbre de décision grâce aux datasets d'entraînement
    """
    model = tree.DecisionTreeClassifier(max_leaf_nodes=10,
                                        min_samples_leaf=10,
                                        min_samples_split=30).fit(X_train, y_train)
    return model

def Evaluation(model, X_test, y_test, isLogit):
    """
    Affiche les résultats du modèle avec les datasets de test
    """
    file = args.filename.split('.')[0].split('/')[-1]
    predict_Y = model.predict(X_test)
    names = model.feature_names_in_
    if isLogit:
        print('Intercept', model.intercept_)
        df_coefficents = pd.DataFrame({'Variables':names,
                                    'Coefficients':model.coef_[0],
                                    'Odd-Ratios':np.exp(model.coef_[0])})
        print(df_coefficents.to_string())
    else:
        df_importance = pd.DataFrame({'Variables':names,
                                      'Importance':model.feature_importances_}).sort_values(by='Importance',
                                                                                            ascending=False, ignore_index=True)
        print(df_importance.loc[df_importance['Importance']!=0])
        tree.plot_tree(model, feature_names= list(X_test.columns) , filled=True)
        plt.savefig(f'images/Tree_{file}.pdf')
    Accuracy = model.score(X_test,y_test)
    class_report = classification_report(y_test, predict_Y, output_dict=True)
    class_report = pd.DataFrame(class_report)
    f1_score = class_report.loc['f1-score']['True']
    MCC = matthews_corrcoef(y_test, predict_Y)
    AUC = ROC(model, X_test, y_test, isLogit)

    df_metrics = pd.DataFrame({'Valeur':[AUC, f1_score, Accuracy, MCC],
                               'Cible':[0.8, 0.75, 0.85, 0.75]},
                               index = ['AUC', 'F1-Score', 'Accuracy', 'MCC'])
    return df_metrics


def ROC(model, X_test, y_test, isLogit):
    """
    """
    if isLogit:
        file = args.filename.split('.')[0].split('/')[-1]+'_Logit'
    else:
        file = args.filename.split('.')[0].split('/')[-1]+'_Tree' 

    FER, TER, threshold = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    AUC = auc(FER,TER)

    ### ROC ####
    plt.figure()
    plt.plot(FER, TER)
    plt.annotate(f'AUC:{round(AUC, 2)}', (0.7,0.3))
    plt.title('Courbe ROC et AUC')
    plt.grid()
    plt.savefig(f'images/ROC_{file}.png')
    # plt.show()

    return AUC

def Scoring(df_metrics, isLogit):
    """
    
    """
    if isLogit:
        model = 'Logit'
    else:
        model = 'Tree'

    print(f"#### SCORING {model} ####")
    print(df_metrics)
    score = df_metrics.apply(lambda row : 1 if row['Valeur'] > row['Cible'] else 0, axis=1).sum()
    print(f'Score du {model}:', score)

def LOGIT(df):
    """
    """
    X_train, X_test, y_train, y_test = Create_Train_Test(df)
    vars = ['native-country_United-States', 'workclass_Private', 'occupation_Prof-specialty', 
            'gender_Male', 'education_HS-grad', 'relationship_Husband', 'marital-status_Married-civ-spouse',
            'race_White']
    refs_vars, X_train, X_test = Drop_References_Variables(X_train, X_test, vars)
    model_logit = Logistic_Regression(X_train, y_train)
    df_metrics_logit = Evaluation(model_logit, X_test, y_test, True)
    Scoring(df_metrics_logit, True)

def TREE(df):
    """
    """
    X_train, X_test, y_train, y_test = Create_Train_Test(df)
    model_decision_tree = DecisionTree(X_train, y_train)
    df_metrics_tree = Evaluation(model_decision_tree, X_test, y_test, False)
    Scoring(df_metrics_tree, False)

def main():

    df = pd.read_csv(args.filename)
    TREE(df)

if __name__ == "__main__":
    main()