import pandas as pd

def delete_na(df):
    """
    Permet de supprimer toutes les lignes contenant au moins un '?' soit une valeur manquante
    Valeur manquantes supprimées : 3620 soit 7,4% des individus
    """
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)

    return df

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
    serie_age = pd.read_csv('files/files_classes_opti/classes_opti_alexander_age.csv')
    serie_hpw = pd.read_csv('files/files_classes_opti/classes_opti_alexander_hours-per-week.csv') 
    df_classes_opti = pd.concat([serie_age, serie_hpw], axis=1)
    df_classes_opti.columns = ['age_opti', 'hours-per-week_opti']
    df = pd.concat([df, df_classes_opti], axis=1)
    df.drop(['age', 'hours-per-week'], inplace=True, axis=1)
    return df

def regroupement(df):
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
        {'Local-gov|State-gov': 'State-Local-gov'}, regex=True)

    df['occupation'] = df['occupation'].replace(
        {'Priv-house-serv|Other-service|Handlers-cleaners': 'Occupation:Very-Low-income',
        'Farming-fishing|Machine-op-inspct|Adm-clerical': "Occupation:Low-income",
        'Transport-moving|Craft-repair' : 'Occupation:Mid-income',
        'Sales|Armed-Forces|Tech-support|Protective-serv|Protective-serv' :'Occupation:High-income',
        'Prof-specialty|Exec-managerial' : 'Occupation:VeryHigh-income'}, regex=True)
    
    return df

def classes_manuelles(df):
    """
    Génère des classes manuelles
    Pour les variables capital-gain et capital-loss
        ->  Devient des variables binaires (pour le moment?): gagne ou ne gagne pas
                                                              perds ou ne perds pas
    """
    serie_gain = df['capital-gain']
    serie_loss = df['capital-loss']

    serie_classes_gain = serie_gain.apply(lambda x : 1 if x!=0 else 0)
    serie_classes_loss = serie_loss.apply(lambda x : 1 if x!=0 else 0)
    df_classes = pd.concat([serie_classes_gain, serie_classes_loss], axis=1)
    df_classes.columns = ['capital-gain-classes', 'capital-loss-classes']
    df = pd.concat([df, df_classes], axis=1)
    df.drop(['capital-gain', 'capital-loss'], inplace=True, axis=1)
    return df

def main_clean():
    """
    Renvoie un dataframe sans valeurs manquantes ni colonnes inutiles
    les outliers sont tjs présents
    """
    df = pd.read_csv('files/revenus.csv')
    df = drop_columns(df)
    df = delete_na(df)
    df.reset_index(inplace=True, drop=True)
    df.to_csv('files/clean.csv', index=False)

def main_clean_classes_V0():
    """
    Renvoie le dataframe clean avec l'application des classes optimisées sur age et sur hours-per-week,
    ainsi que les classes manuelles pour capital-gain et capital-loss
    """
    df = pd.read_csv('files/clean.csv')
    df = apply_opti_classes(df)
    df = classes_manuelles(df)
    df.to_csv('files/clean_classes_V0.csv', index=False)

def main_clean_classes_V1():
    """
    """
    df = pd.read_csv('files/clean.csv')
    df = apply_opti_classes(df)
    df.to_csv('files/clean_classes_V1.csv', index=False)

def main_clean_classes_V2():
    """
    """
    df = pd.read_csv('files/clean.csv')
    df = apply_opti_classes(df)
    df = regroupement(df)
    df.to_csv('files/clean_classes_V2.csv', index=False)

main_clean_classes_V1()
