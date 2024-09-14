import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import time

def generate_classes(serie):
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
        df_classes = pd.cut(serie, random_cuts(serie, 4), right=True, include_lowest=True)
        while any(df_classes.value_counts()<(0.1*len(df_classes))):

            df_classes = pd.cut(serie, random_cuts(serie, 4), right=True, include_lowest=True)

        return df_classes
        
    df_classes = create_test_classes(serie)
    return df_classes

def optimize_classes(serie, df_income):
    """
    """
    def khi2(df_classes, df_income):
        """
        """
        table = pd.crosstab(df_classes, df_income)
        khi2, pval, ddl, contigent_theo = chi2_contingency(table)
        return khi2, pval, ddl
    
    df_classes_min = generate_classes(serie)
    khisq_min, pval_min, ddl_min = khi2(df_classes_min, df_income)
    n=100
    deltas=[]

    for i in range(n):
        time1 = time.time()*1000
        df_classes = generate_classes(serie)
        time2 = time.time()*1000
        deltas.append((time2-time1)/1000)
        print(i,'  Temps:', (deltas[-1]))
        khisq, pval, ddl = khi2(df_classes, df_income)
        if (khisq > khisq_min) and (pval<=pval_min):
            conv = i
            df_classes_min = df_classes
            khisq_min = khisq
            pval_min = pval
            ddl_min = ddl
    sum_delta = sum(deltas)
    print('Temps total:', sum_delta, sum_delta/60)
    print('Temps moyen pour une génération de table:', sum_delta/n)
        
    return df_classes_min, khisq_min, pval_min, ddl_min, conv
        
def main():
    df = pd.read_csv('files/clean.csv')
    df_income = df['income']
    df_classes, khisq, pval, ddl, conv = optimize_classes(df['hours-per-week'], df_income)
    print(df_classes, khisq, pval, ddl, conv)
    df_classes.to_csv(f'files/classes_opti_{df_classes.name}.csv', index=False)


if __name__ == "__main__":
    main()
