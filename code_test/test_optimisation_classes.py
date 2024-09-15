import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import time
from collections import Counter
import warnings

class GeneratorClasses:

    def __init__(self, serie, bins):
        self.serie = serie
        self.bins = bins
        self.effectif = len(serie)
        self.count_population_left = self.effectif
        self.count = 0

    def generate_counter_items(self, min_level):
        """
        """
        serie = self.serie.loc[self.serie>=min_level]
        counter = Counter(serie)
        for i in sorted(counter.items()):
            yield i

    def generate_classes_for_levels(self, min_level, count_population_left, count, bin):
        """
        """
        min_threshold = int(0.1*self.effectif)
        # print(min_threshold)
        max_threshold = int(count_population_left - (self.bins - bin)*min_threshold)
        # print(max_threshold)
        df_classes = pd.DataFrame()
        if bin != self.bins:
            while min_threshold <= max_threshold:
                sum = 0
                self.count = count
                classes = []
                self.count_population_left = count_population_left
                # print(self.count_population_left)
                for i in self.generate_counter_items(min_level):
                    classes.append(i[0])
                    sum += i[1]
                    if min_threshold < sum:
                        self.count_population_left -= sum
                        self.count+=sum
                        # if bin == 3:
                            # print(i)
                            # print('bin', bin, '; lvl:', classes[-1], '; count:', self.count, '; min_th:', min_threshold, '; max_th:', max_threshold, '; popleft', self.count_population_left)
                            # if classes[-1]==50 and count_population_left <= 35689:
                            #     raise Exception
                        df_classes = pd.concat([df_classes, pd.DataFrame({'from':[classes[0]], 'to':[classes[-1]], 'to+1':[classes[-1]+1], 'Count': [self.count], 'Pop left':[self.count_population_left]})], ignore_index=True)
                        min_threshold = sum 
                        break
            df_classes.drop(df_classes.tail(1).index, inplace=True)
        else:
            df_classes = pd.concat([df_classes, pd.DataFrame({'from':[min_level], 'to':[self.serie.max()], 'to+1':[self.serie.max()+1],  'Count': [count+count_population_left], 'Pop left':[0]})], ignore_index=True)
       
        return df_classes
    
    

def test():
    df = pd.read_csv('../files/clean.csv')
    gen = GeneratorClasses(df['age'], 5)

    print(gen.generate_classes_for_levels(27, 35689, 3))

    

    

def main():
    df = pd.read_csv('../files/clean.csv')
    gen = GeneratorClasses(df['age'], 4)

    dfs = [gen.generate_classes_for_levels(gen.serie.min(), gen.effectif, 0, 1)]
    print(dfs[0])
    classes = [f'x{i}' for i in range(gen.bins+1)]
    print(classes)
    df_all_combinaisons = pd.DataFrame(columns=classes)
    for bin in range(2, gen.bins+1):
        print('BIN', bin)
        for df in dfs:
            if bin == 3:
                df.apply(lambda row : print(row['from'], row['to'], row['to+1'], row['Pop left']), axis=1)
            dfs = df.apply(lambda row : gen.generate_classes_for_levels(row['to+1'], row['Pop left'], row['Count'], bin), axis=1)
            
main()