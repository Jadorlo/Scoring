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
    print(f'Temps total: {sum_delta}sec, {sum_delta/60}min')
    print(f'Temps moyen pour une génération de table: {sum_delta/n}sec/indiv.')
        
    return df_classes_min, khisq_min, pval_min, ddl_min, conv

def test(df_classes, khisq, pval, df_income, col):
    df1 = pd.read_csv(f'files/classes_opti/classes_opti_alexander_{col}.csv')
    serie_classe_actuelle = df1['classes_opti']
    table_actuelle = pd.crosstab(serie_classe_actuelle, df_income)
    khi_act, pval_act, ddl, contigent_theo = chi2_contingency(table_actuelle)
    if (khisq > khi_act) and (pval<=pval_act):
        df_classes.name = 'classes_opti' 
        df_classes.to_csv(f'files/classes_opti/classes_opti_alexander_{col}.csv', index=False)
        return True
    else:
        return False

        
def main():
    df = pd.read_csv('files/revenus.csv')
    df_income = df['income']
    col = 'hours-per-week'
    df_classes, khisq, pval, ddl, conv = optimize_classes(df[col], df_income, 4)
    print(df_classes, khisq, pval, ddl, conv)
    print(test(df_classes, khisq, pval, df_income, col))


if __name__ == "__main__":
    main()
