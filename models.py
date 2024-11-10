import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve, matthews_corrcoef
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats as stat


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

def Drop_References_Variables(X_train, X_test, vars):
    """
    Drop les variables de référence pour le modèle Logit
    """
    refs_vars = X_train[vars].copy()
    X_train.drop(vars, axis=1, inplace=True)
    X_test.drop(vars, axis=1, inplace=True)

    return refs_vars, X_train, X_test

def Logistic_Regression(X_train, y_train):
    """
    Crée le Logit grâce aux datasets d'entraînement
    """
    model = LogisticRegression(random_state = 0, fit_intercept=True).fit(X_train, y_train)

    return model

def DecisionTree(X_train, y_train):
    """
    Créer l'arbre de décision grâce aux datasets d'entraînement
    """
    # model = tree.DecisionTreeClassifier(max_depth=13,
    #                                     max_leaf_nodes=63,
    #                                     min_samples_leaf=20, 
    #                                     min_samples_split=60).fit(X_train, y_train)

    model = tree.DecisionTreeClassifier().fit(X_train, y_train)
    return model

def GrilleRecherche(X_train, X_test, y_train, y_test):
    """
    """
    model = tree.DecisionTreeClassifier(random_state=42)
    # parameters = {'ccp_alpha':np.linspace(0,0.01, 10),
    #               'criterion': ['gini', 'entropy'],
    #               'max_depth': list(range(7, 14)),
    #               'max_leaf_nodes':np.linspace(40, 60, 5, dtype = int),
    #               'min_impurity_decrease': np.linspace(0,0.5,5),
    #               'min_samples_split':np.linspace(20,60,5, dtype = int)
    #               }

    parameters = {'max_depth': list(range(7, 14)),
                  'max_leaf_nodes':np.linspace(40, 70, 10, dtype = int),
                  'min_samples_split':np.linspace(30, 60, 10, dtype = int),
                  'min_samples_leaf':np.linspace(20, 40, 10, dtype = int)
                  }
    
    clf = GridSearchCV(model, parameters, scoring='accuracy', n_jobs=5)
    clf.fit(X_train, y_train)

    print(clf.best_params_)

    results = clf.cv_results_

    # Tri des résultats par la performance (score moyen dans les cross-validations)
    sorted_results = sorted(
    zip(results['mean_test_score'], results['params']),
    key=lambda x: x[0],
    reverse=True)

    # Sélection des 10 meilleures combinaisons
    top_10_combinations = sorted_results[:10]

    # Affichage des 10 meilleures combinaisons
    for rank, (score, params) in enumerate(top_10_combinations, start=1):
        print(f"Rank {rank}: Score = {score:.4f}, Parameters = {params} \n")
    

def Evaluation(model, X_test, y_test, isLogit):
    """
    Affiche les résultats du modèle avec les datasets de test
    Différenciation entre Logit et arbre de décision
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
        criterion = model.get_params()['criterion']
        df_importance = pd.DataFrame({'Variables':names,
                                      'Importance':model.feature_importances_}).sort_values(by='Importance',
                                                                                            ascending=False, ignore_index=True)
        print(df_importance.loc[df_importance['Importance']!=0])
        tree.plot_tree(model, feature_names= list(X_test.columns), filled=True)
        plt.savefig(f'images/Trees/Tree_{criterion}_{file}.pdf')

    print('Matrice de confusion:\n', confusion_matrix(y_test, predict_Y))
    Accuracy = model.score(X_test,y_test)
    class_report = classification_report(y_test, predict_Y, output_dict=True)
    class_report = pd.DataFrame(class_report)
    print(class_report)
    f1_score = class_report.loc['f1-score']['accuracy']
    MCC = matthews_corrcoef(y_test, predict_Y)
    AUC = ROC(model, X_test, y_test, isLogit)

    #Calcul du gAUC moyen
    gAUC_liste = []
    for var in df.select_dtypes(include='object').columns:
        print(var)
        gauc = gAUC(model, X_test, y_test, var)
        gAUC_liste.append(gauc)
    gauc = np.mean(gAUC_liste)

    df_metrics = pd.DataFrame({'Valeur':[AUC, f1_score, Accuracy, MCC, gauc],
                               'Cible':[0.8, 0.75, 0.85, 0.75, 0.8]},
                               index = ['AUC', 'F1-Score', 'Accuracy', 'MCC', 'gAUC'])
    return df_metrics


def ROC(model, X_test, y_test, isLogit):
    """
    Charge la courbe roc du modèle et retourne l'AUC
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
    plt.savefig(f'images/ROC/ROC_{file}.png')
    # plt.show()

    return AUC

def gAUC(model, X_test, y_test, var):
    """
    Calcule le gAUC pour une variable qualitative arbitraire
    """
    X_test.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)
    df_var = X_test.filter(regex=f"^{var}")
    
    auc_liste = []

    for col in df_var:
        print(col)
        X_auc = X_test.loc[df_var[col]==True]
        y_auc = y_test.iloc[X_auc.index]
        y_prob = model.predict_proba(X_auc)[:,1]

        col_FER, col_TER, threshold = roc_curve(y_auc, y_prob)
        AUC = auc(col_FER,col_TER)
        print(AUC)
        auc_liste.append(AUC)
    gAUC = np.mean(auc_liste)
    return gAUC
    

def Scoring(df_metrics, isLogit):
    """
    Calcul le score du modèle en fonction des métriques de df_metrics
    """
    if isLogit:
        model_name = 'Logit'
    else:
        model_name = 'Tree'

    print(f"#### SCORING {model_name} ####")
    print(df_metrics)
    score = df_metrics.apply(lambda row : 1 if row['Valeur'] > row['Cible'] else 0, axis=1).sum()
    print(f'Score du {model_name}:', score)
    return df_metrics, score

def Tracking_Dataframe(params, df_metrics, score, isLogit):
    """
    """
    now = datetime.now()
    
    df_tracking = pd.DataFrame([{'Date':now.strftime("%d/%m/%Y %H:%M:%S"), 'Score':score, 'File':args.filename.split('/')[-1].split('.')[0]}])
    df_tracking[df_metrics.index] = df_metrics['Valeur'].T
    df_cible = df_metrics['Cible'].T
    df_tracking[[f'{i}_cible' for i in df_metrics.index.tolist()]] = df_cible
    df_tracking_columns = df_tracking.columns
    df_params = pd.DataFrame([params], columns=list(params.keys()))
    df_tracking = pd.concat([df_tracking, df_params], axis=1)
    new_index = df_tracking_columns.insert(1, df_params.columns)
    df_tracking = df_tracking.reindex(columns=new_index)
    df_tracking.index.name = "index"
    if not isLogit:
        df_tracking.to_csv('files/tracking_models_files/tracking_decision_tree.csv', mode='a', header=False)
    else:
        df_tracking.to_csv('files/tracking_models_files/tracking_logit.csv', mode='a', header=False)
    return df_tracking

def LOGIT(df):
    """
    Réunion des fonctions nécessaires au fonctionnement du modèle LOGIT
    """
    X_train, X_test, y_train, y_test = Create_Train_Test(df)
    print(X_train.dtypes)
    #vars = ['native-country_United-States', 'workclass_Private', 'occupation_Occupation:Mid-income', 
    #        'gender_Male', 'education_HS-grad', 'relationship_Husband', 'marital-status_Married-civ-spouse',
    #        'race_White', 'age_(28.0, 33.0]', 'hours-per-week_(37.0, 43.0]']
    #refs_vars, X_train, X_test = Drop_References_Variables(X_train, X_test, vars)
    model_logit = Logistic_Regression(X_train, y_train)
    df_metrics_logit = Evaluation(model_logit, X_test, y_test, True)
    df_metrics_logit, score = Scoring(df_metrics_logit, True)
    df_tracking = Tracking_Dataframe(model_logit.get_params(), df_metrics_logit, score, True)

def TREE(df):
    """
    Réunion des fonctions nécessaires au fonctionnement du modèle Arbre de Décision
    """
    X_train, X_test, y_train, y_test = Create_Train_Test(df)
    model_decision_tree = DecisionTree(X_train, y_train)
    df_metrics_tree = Evaluation(model_decision_tree, X_test, y_test, False)
    df_metrics_tree, score = Scoring(df_metrics_tree, False)
    df_tracking = Tracking_Dataframe(model_decision_tree.get_params(), df_metrics_tree, score, False)

def TestGridSearch(df):
    """
    """
    X_train, X_test, y_train, y_test = Create_Train_Test(df)
    GrilleRecherche(X_train, X_test, y_train, y_test)

def main():
    global df
    df = pd.read_csv(args.filename)
    LOGIT(df)

if __name__ == "__main__":
    main()