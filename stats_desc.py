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
    #print(mean, std)

    normalized_df = (df[col]-mean)/std
    X0 = normalized_df.to_list()
    #print(np.mean(X0), np.std(X0))

    X1 = np.linspace(np.min(X0), np.max(X0), len(X0))
    
    borne_minX0, borne_maxX0= mean-3*std, mean+3*std
    borne_minX1, borne_maxX1= np.mean(X0)-3*np.std(X0), np.mean(X0)+3*np.std(X0)

    #print(borne_min, borne_max)
    Y1 = []
    for x in X1:
        Y1.append(1/(np.sqrt(2*np.pi))*np.exp(-1/2*x**2))

    is_in = ['blue' if borne_minX1 <= x <= borne_maxX1 else 'red' for x in X1]
    nb_out = sum(map(lambda x: (x<borne_minX0)|(x>borne_maxX0), X0))
    prc_out = nb_out/len(X0)

    print(pd.DataFrame({'X':X0, 'IN':is_in}))

    plt.title(col.upper())
    plt.grid()
    plt.scatter(X1, Y1, c=is_in, alpha=0.25, s=1)
    plt.annotate(f'> 3$\sigma$ \n Nb={nb_out} \n %={round(prc_out, 2)*100}', xy=((np.max(X1)+2.5)/2,0.25), color="red")
    plt.vlines(3, ymin=-0.1, ymax=0.5, color='black')
    plt.ylim((-0.1,0.41))
    plt.savefig(f'images/zscore_density_{col}.pdf')
    plt.show()

def zscore2(serie, delete):
    """
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
    plt.savefig(f'images/zscore_density_{serie.name}.pdf')
    plt.show()


def main():
    df = pd.read_csv(args.filename)
    # boxplot(df)
    # pieplot(df)
    # histogram(df)
    for col in ['age', 'hours-per-week', 'capital-gain', 'capital-loss']:
        zscore2(df[col], False)
    #zscore2(df['age'], False)


if __name__ == "__main__":
    main()