import pandas as pd

def delete_na(df):
    """
    Permet de supprimer toutes les lignes contenant au moins un '?' soit une valeur manquante
    Valeur manquantes supprimées : 3620 soit 7,4% des individus
    """
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)

    return df

def delete_neverworked(df):
    """
    """
    df_nw =df.drop(df.loc[df['workclass']=='Never-worked'].index, axis=0)
    return df_nw

def drop_columns(df):
    """
    Drop les colonnes fnlwgt, educational-nul car inutiles
    """
    df.drop(["fnlwgt", "educational-num"], inplace=True, axis=1)
    return df

def apply_opti_classes(df):
    """
    Applique les classes optimisées générées par le fichier optimisation_classes.py
    """
    serie_age = pd.read_csv('files/classes_opti/classes_opti_alexander_age.csv')
    serie_hpw = pd.read_csv('files/classes_opti/classes_opti_alexander_hours-per-week.csv') 
    df_classes_opti = pd.concat([serie_age, serie_hpw], axis=1)
    df.drop(['age', 'hours-per-week'], inplace=True, axis=1)
    df_classes_opti.columns = ['age', 'hours-per-week']
    df = pd.concat([df, df_classes_opti], axis=1)
    return df

def regroupement_V1(df):
    """
    """
    # Regroupement des individus Never-Worked (10/48843), Without-pay(21/48843) et ? dans No_income_or_unknown
    df['workclass'] = df['workclass'].replace(
        {'\?' : 'No_income_or_unknown'}, regex=True)
    
    # Regroupement des autres nationalités et ? dans Other_country
    df['native-country'] = df['native-country'].replace(
        {'\?': 'Other_country'}, regex=True)
    
    # Remplacement des ? dans Other-service 
    df['occupation'] = df['occupation'].replace(
        {'\?' : 'Other-service'}, regex=True)
    
    return df
    
def regroupement_V2(df):
    """
    """
    # Regroupement des personnes d'ethnies autre que blanc et noir en "Other_race"
    df['race'] = df['race'].replace(
        {'Asian-Pac-Islander|Amer-Indian-Eskimo|Other': 'Other_race'}, regex=True)

    # Regroupement des personnes d'origines différentes en une cartégorie "Other"
    df['native-country'] = df['native-country'].replace(
        {'^(?!United-States$).+': 'Other_country'}, regex=True)

    # Regroupement des personnes mariés mais avec un conjoint dans l'armé avec les personnes mariés mais avec un conjoint absent pour diverse raisons
    df['marital-status'] = df['marital-status'].replace(
        {'Married-AF-spouse|Divorced|Widowed|Separated|Married-spouse-absent': 'Alone'}, regex = True)

    df['education'] = df['education'].replace(
        {'Preschool|1st-4th|5th-6th|7th-8th|9th|10th|11th|12th': 'Low-education',
        'Doctorate|Prof-school': "Graduation",
        "Assoc-acdm|Assoc-voc" : 'Assoc'}, regex=True)

    df['workclass'] = df['workclass'].replace(
        {'Local-gov|State-gov': 'State-Local-gov',
         'Never-worked|Without-pay|\?' : 'No_income_or_unknown'}, regex=True)

    df['occupation'] = df['occupation'].replace(
        {'\?|Priv-house-serv|Other-service|Handlers-cleaners': 'Occupation:Very-Low-income',
        'Farming-fishing|Machine-op-inspct|Adm-clerical': "Occupation:Low-income",
        'Transport-moving|Craft-repair' : 'Occupation:Mid-income',
        'Sales|Armed-Forces|Tech-support|Protective-serv|Protective-serv' :'Occupation:High-income',
        'Prof-specialty|Exec-managerial' : 'Occupation:Very-High-income'}, regex=True)
    
    return df

def main_data_V1():
    """
    Crée le fichier de données V2 avec le regroupement des ?, et des modalités aberrantes (trop peu représentées)
    soit Never-worked, Without-pay et les nationalités différentes de United-States.
    """
    df = pd.read_csv('files/revenus.csv')
    df = drop_columns(df)
    df = regroupement_V1(df)
    df.to_csv('files/data/data_V1.csv', index=False)

def main_data_V2():
    """
    Crée le fichier de données V2 avec le regroupement des ?, le regroupement des modalités 
    et l'application des classes optimisées pour les variables age et hours-per-week
    """
    df = pd.read_csv('files/revenus.csv')
    df = drop_columns(df)
    df = apply_opti_classes(df)
    df = regroupement_V2(df)
    df.to_csv('files/data/data_V2.csv', index=False)

main_data_V1()
main_data_V2()