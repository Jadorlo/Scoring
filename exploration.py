import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def drop_columns(df):
    """
    Drop la colonne fnlwgt car inutile
    """
    df.drop("fnlwgt", inplace=True, axis=1)
    return df

def delete_na(df):
    """
    Permet de supprimer toutes les lignes contenant au moins un '?' soit une valeur manquante
    Valeur manquantes supprimées : 3620 soit 7,4% des individus
    """
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    return df

def main():
    df = pd.read_csv('revenus.csv')
    df = drop_columns(df)
    df = delete_na(df)
    #boxplot(df)
    pieplot(df)

main()

