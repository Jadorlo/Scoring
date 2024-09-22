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
    Drop la colonne fnlwgt car inutile
    """
    df.drop(["fnlwgt", "educational-num"], inplace=True, axis=1)

    return df

def apply_classes(df):

    serie_age = pd.read_csv('files/files_classes_opti/classes_opti_alexander_age.csv')
    serie_hpw = pd.read_csv('files/files_classes_opti/classes_opti_alexander_hours-per-week.csv') 
    df_classes_opti = pd.concat([serie_age, serie_hpw], axis=1)
    df_classes_opti.columns = ['age_opti', 'hours-per-week_opti']
    print(df_classes_opti)
    df = pd.concat([df, df_classes_opti], axis=1)
    df.drop(['age', 'hours-per-week'], inplace=True, axis=1)
    return df



def main_clean():
    """
    Renvoie un dataframe sans valeurs manquantes ni colonnes inutiles
    """
    df = pd.read_csv('files/revenus.csv')
    df = drop_columns(df)
    df = delete_na(df)
    df.reset_index(inplace=True, drop=True)
    df.to_csv('files/clean.csv', index=False)

def main_clean_classes():
    """
    Renvoie le dataframe clean avec l'application des classes optimiser sur age et sur hours-per-week
    """
    df = pd.read_csv('files/clean.csv')
    apply_classes(df)

main_clean_classes()