import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import numpy as np
from scipy.stats import norm


parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

def boxplot(df):
    """
    
    """
    plt.figure()
    df.plot(kind='box', subplots=True, figsize=(22,6))
    file = args.filename.split('.')[0].split('/')[-1]
    plt.savefig(f'images/boxplot_{file}.png')

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
    file = args.filename.split('.')[0].split('/')[-1]
    plt.savefig(f'images/piecharts_{file}.png')

def histogram(df):
    """
    """
    columns = df.select_dtypes(include='int64').columns
    fig, axs = plt.subplots(2, 2, figsize=(15,10))
    for k, col in enumerate(columns):
        axs[k%2, k//2].hist(df[col], bins=10)
        axs[k%2, k//2].set_title(col.upper())
    
    plt.tight_layout()
    fig.suptitle('Dsitribution des variables qualitatives')
    file = args.filename.split('.')[0].split('/')[-1]
    plt.savefig(f'images/histogram_{file}.png')
    plt.show()

def zscore(df, col, delete:bool):
    """
    
    """

    mean = df[col].mean()
    std = df[col].std()

    normalized_df = (df[col]-mean)/std
    normalized_list = normalized_df.to_list()

    mean_norm = normalized_df.mean()
    std_norm = normalized_df.std()
    min = normalized_df.min()
    max = normalized_df.max()

    print(min, max)
    X1 = np.linspace(min, max, 45222)
    Y1 = 1/np.sqrt(2*np.pi)*np.exp(-1/2*((X1-mean_norm)/std_norm)**2)
    
    #Y2 = norm.pdf(normalized_list, mean, std)
    
    borne_min, borne_max= mean-3*std, mean+3*std
    is_in = ['blue' if borne_min <= x <= borne_max else 'red' for x in normalized_list]


    x = np.linspace(mean - 4 * std, mean + 4 * std, 1000)
    x = (x-x.mean())/x.std()
    y = norm.pdf(x, x.mean(), x.std())


    plt.figure(figsize=(10, 6))
    #plt.plot(x, y, label='Densité', color='black')
    plt.vlines(3, ymin =-0.11, ymax=0.46, color='red')
    plt.scatter(normalized_list, Y1, alpha=0.1)
    plt.ylim((0, 0.4))
    plt.show()
    
    

    



def main():
    df = pd.read_csv(args.filename)
    # boxplot(df)
    # pieplot(df)
    # histogram(df)
    zscore(df, 'age', False)

if __name__ == "__main__":
    main()