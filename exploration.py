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

def boxplot(df):
    plt.figure()
    df.plot(kind='box', subplots=True, figsize=(22,6))
    plt.show()

def pieplot(df):
    plt.figure()
    df.select_dtypes(include='object').plot(kind='pie',subplots=True, figsize=(22,6))
    plt.show()

def test():
    pass

def main():
    df = pd.read_csv('/Users/alexanderlunel/Documents/LILLE/Master/MasterSIAD/M2/Scoring/Etude de cas/revenus.csv')
    df = drop_columns(df)
    df = delete_na(df)
    #boxplot(df)
    pieplot(df)

main()

