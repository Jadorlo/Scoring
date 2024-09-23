import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import numpy as np
from scipy.stats import norm


parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

#plt.rcParams['text.usetex'] = True

def boxplot(df):
    """
    Affiche les boxplots des variables quantitatives
    """
    plt.figure()
    df.plot(kind='box', subplots=True, figsize=(22,6))
    file = args.filename.split('.')[0].split('/')[-1]
    plt.savefig(f'images/boxplot_{file}.png')

def pieplot(df):
    """
    Affiche les proportions des classes des variables qualitatives
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
    Affiche les histogrammes des variables quantitatives
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

def zscore(serie, delete):
    """
    Affiche la loi normale de la distribution de la variable.
    Indique quels individus seraient supprimés en appliquant le Zscore
    """
    mean = serie.mean()
    std = serie.std()
    X0 = (serie-mean)/std #Variable normalisée
    borne_minX0 = -3
    borne_maxX0 = 3
    df_master = pd.concat([serie, X0], axis=1)
    df_master.columns = ['age', 'X0']
    df_master['Y0'] = df_master['X0'].apply(lambda x : 1/(np.sqrt(2*np.pi))*np.exp(-1/2*x**2))
    df_master['Is_inX0'] = df_master['X0'].apply(lambda x : 1 if borne_minX0 <= x <= borne_maxX0 else 0)
    df_master['Is_inX0color'] = df_master['Is_inX0'].apply(lambda x : 'blue' if x==1 else 'red')
    nb_out = len(df_master['Is_inX0']) - df_master['Is_inX0'].sum()
    prc_out = nb_out/len(df_master)
    
    #print(df_master.loc[df_master['X0']>3])
   
    plt.figure()
    plt.title(label=serie.name.upper())
    plt.grid()
    plt.scatter(df_master['X0'], df_master['Y0'], c=df_master['Is_inX0color'], alpha=0.25, s=1)
    plt.annotate(f'x<-3$\sigma$\n ou x>3$\sigma$ \n Nb={nb_out} \n %={round(prc_out, 3)*100}', xy=((df_master['X0'].max()+2.5)/2,0.25), color="red")
    plt.vlines(3, ymin=-0.1, ymax=0.5, color='black')
    plt.ylim((-0.1,0.41))
    plt.savefig(f'images/Z-Scores/zscore_density_{serie.name}.pdf')
    plt.show()

def V_Cramer_matrix(df):
    """
    """
    pass

def capital_without_0(df):
    """
    """
    serie_gain = df['capital-gain'].loc[df['capital-gain']>0]
    serie_loss = df['capital-loss'].loc[df['capital-loss']>0]
    print(len(serie_gain))
    print(len(serie_loss))
    
    fig, axs = plt.subplots(2, 1, figsize=(15,10))
    
    axs[1].hist(serie_gain, bins=10)
    axs[1].set_title(serie_gain.name.upper())
    
    axs[0].hist(serie_loss, bins=10)
    axs[0].set_title(serie_loss.name.upper())

    plt.tight_layout()
    fig.suptitle('Dsitribution des variables qualitatives')
    file = args.filename.split('.')[0].split('/')[-1]
    #plt.savefig(f'images/histogram_{file}.png')
    plt.show()


def main():
    df = pd.read_csv(args.filename)
    # boxplot(df)
    # pieplot(df)
    # histogram(df)
    # for col in ['age', 'hours-per-week', 'capital-gain', 'capital-loss']:
    #     zscore(df[col], False)
    capital_without_0(df)


if __name__ == "__main__":
    main()