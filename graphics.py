import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def boxplot(df):
    """
    
    """
    df_box = df.drop('index', axis=1)
    plt.figure()
    df_box.plot(kind='box', subplots=True, figsize=(22,6))
    plt.savefig('images/boxplot.png')

def pieplot(df):
    """
    
    """
    columns = df.select_dtypes(include='object').columns
    fig, axs = plt.subplots(3, 3, figsize=(15,15))
    for k, col in enumerate(columns):
        df_prop = df[col].value_counts()
        axs[k%3, k//3].pie(x=df_prop, labels = df_prop.index, autopct='%1.1f%%', labeldistance=1.1)
        axs[k%3, k//3].set_title(col.upper())
    
    plt.tight_layout()
    fig.suptitle('Proportion des variables qualitatives')
    plt.savefig('images/piecharts.png')    

def main():
    df = pd.read_csv('clean.csv')
    boxplot(df)
    pieplot(df)

if __name__ == "__main__":
    main()