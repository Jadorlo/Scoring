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

def main():
    """
    
    """
    df = pd.read_csv('files/revenus.csv')
    df = drop_columns(df)
    df = delete_na(df)
    df.index.name = 'index'
    df.to_csv('files/clean.csv')

main()