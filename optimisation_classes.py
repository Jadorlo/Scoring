import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import time

def generate_classes(serie, bins):
    """
    """

    def random_cuts(serie, bins):
        """
        """
        serie_min, serie_max = serie.min(), serie.max()
        cuts = np.sort([np.random.choice(range(serie_min+1, serie_max-1, 1), size=bins, replace=False)])
        cuts = np.insert(cuts, 0, serie_min)
        cuts = np.insert(cuts, -1, serie_max)
        cuts = np.sort(cuts)
        return cuts
    
    def create_test_classes(serie):
        """
        """
        df_classes = pd.cut(serie, random_cuts(serie, bins), right=True, include_lowest=True)
        while any(df_classes.value_counts()<(0.1*len(df_classes))):

            df_classes = pd.cut(serie, random_cuts(serie, bins), right=True, include_lowest=True)

        return df_classes
        
    df_classes = create_test_classes(serie)
    return df_classes



def optimize_classes(serie, df_income, bins):
    """
    """
    def khi2(df_classes, df_income):
        """
        """
        table = pd.crosstab(df_classes, df_income)
        khi2, pval, ddl, contigent_theo = chi2_contingency(table)
        return khi2, pval, ddl
    
    df_classes_min = generate_classes(serie, bins)
    khisq_min, pval_min, ddl_min = khi2(df_classes_min, df_income)
    n=100
    deltas=[]

    for i in range(n):
        time1 = time.time()*1000
        df_classes = generate_classes(serie, bins)
        time2 = time.time()*1000
        deltas.append((time2-time1)/1000)
        # print(i,'  Temps:', (deltas[-1]))
        khisq, pval, ddl = khi2(df_classes, df_income)
        if (khisq > khisq_min) and (pval<=pval_min):
            conv = i
            print(conv)
            df_classes_min = df_classes
            khisq_min = khisq
            pval_min = pval
            ddl_min = ddl
    sum_delta = sum(deltas)
    print('Temps total:', sum_delta, sum_delta/60)
    print('Temps moyen pour une génération de table:', sum_delta/n)
        
    return df_classes_min, khisq_min, pval_min, ddl_min, conv

def test(khisq, pval, df_income):
    df0 = pd.read_csv('files/clean.csv')
    df1 = pd.read_csv('files/files_classes_opti/classes_opti_alexander_age.csv')
    serie_originale = df0['age']
    serie_classe_actuelle = df1['age']
    table_actuelle = pd.crosstab(serie_classe_actuelle, df_income)
    khi_act, pval_act, ddl, contigent_theo = chi2_contingency(table_actuelle)
    if (khisq > khi_act) and (pval<=pval_act):
        df = pd.concat([serie_classe_actuelle,serie_originale], axis=1)
        df.to_csv(f'files/files_classes_opti/classes_opti_alexander_{serie_classe_actuelle.name}.csv', columns=['classe_age', 'age_quanti'], index=False)
        return True
    else:
        return False

        
def main():
    df = pd.read_csv('files/clean.csv')
    df_income = df['income']
    df_classes, khisq, pval, ddl, conv = optimize_classes(df['age'], df_income, 5)
    print(df_classes, khisq, pval, ddl, conv)
    print(test(khisq, pval, df_income))


if __name__ == "__main__":
    main()
