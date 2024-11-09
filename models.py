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

    # denom = (2.0*(1.0+np.cosh(model.decision_function(X_train))))

    # if model.get_params()['fit_intercept']:
    #         X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

    # denom = np.tile(denom,(X_train.shape[1],1)).T
    # F_ij = np.dot((X_train/denom).T,X_train) ## Fisher Information Matrix
    # Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix

    # if model.get_params()['fit_intercept']:
    #     model.coef = np.column_stack((model.intercept_, model.coef_))
    # else:
    #     model.coef = model.coef_

    # sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
    # z_scores = model.coef_[0]/sigma_estimates # z-score for eaach model coefficient
    # p_values = [stat.norm.sf(abs(x))*2 for x in z_scores]
    print("p_values")
    
    return model

def DecisionTree(X_train, y_train):
    """
    Créer l'arbre de décision grâce aux datasets d'entraînement
    """
    model = tree.DecisionTreeClassifier(ccp_alpha=0.0001 ,criterion='gini', max_leaf_nodes=30).fit(X_train, y_train)
    return model

def GrilleRecherche(X_train, X_test, y_train, y_test):
    """
    """
    model = tree.DecisionTreeClassifier(random_state=42)
    parameters = {'ccp_alpha':np.linspace(0,0.01, 10),
                  'criterion': ['gini', 'entropy'],
                  'max_depth': list(range(5, 14)),
                  'max_leaf_nodes':np.linspace(10, 60, 10),
                  'min_impurity_decrease': np.linspace(0,0.5,10),
                  'min_samples_split':np.linspace(20,92,10)
                  }
    
    clf = GridSearchCV(model, parameters)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    print(X_train, y_train)
    print(X_test, y_test)

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
    f1_score = class_report.loc['f1-score']['True']
    MCC = matthews_corrcoef(y_test, predict_Y)
    AUC = ROC(model, X_test, y_test, isLogit)

    df_metrics = pd.DataFrame({'Valeur':[AUC, f1_score, Accuracy, MCC, None],
                               'Cible':[0.8, 0.75, 0.85, 0.75, None]},
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
    #print([f'{i}_cible' for i in df_metrics.index.tolist()])
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
        pass
        df_tracking.to_csv('files/tracking_models_files/tracking_decision_tree.csv', mode='a', header=False)
    else:
        df_tracking.to_csv('files/tracking_models_files/tracking_logit.csv', mode='a', header=False)
        pass
    return df_tracking




def LOGIT(df):
    """
    Réunion des fonctions nécessaires au fonctionnement du modèle LOGIT
    """
    X_train, X_test, y_train, y_test = Create_Train_Test(df)
    vars = ['native-country_United-States', 'workclass_Private', 'occupation_Occupation:Mid-income', 
            'gender_Male', 'education_HS-grad', 'relationship_Husband', 'marital-status_Married-civ-spouse',
            'race_White', 'age_opti_(28.0, 35.0]', 'hours-per-week_opti_(39.0, 43.0]']
    refs_vars, X_train, X_test = Drop_References_Variables(X_train, X_test, vars)
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

    df = pd.read_csv(args.filename)
    TestGridSearch(df)

if __name__ == "__main__":
    main()