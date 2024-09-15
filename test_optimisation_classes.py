import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import time
from collections import Counter

class GeneratorClasses:

    def __init__(self, serie, bins):
        self.serie = serie
        self.bins = bins
        self.effectif = len(serie)
        self.count_population_left = self.effectif

    def generate_counter_items(self, min_level):
        """

        """
        self.serie = self.serie.loc[self.serie>=min_level]
        counter = Counter(self.serie)
        for i in sorted(counter.items()):
            yield i

    def generate_classes_for_levels(self, min_level, count_population_left, bin):
        """
        """
        min_threshold = int(0.1*self.effectif)
        # print(min_threshold)
        max_threshold = int(count_population_left - (self.bins - bin)*min_threshold)
        # print(max_threshold)
        df_classes = pd.DataFrame(columns=['Classes', 'Count'])
        if bin != self.bins:
            while min_threshold <= max_threshold:
                sum = 0
                classes = []
                self.count_population_left = count_population_left
                # print(self.count_population_left)
                for i in self.generate_counter_items(min_level):
                    classes.append(i[0])
                    sum += i[1]
                    print('lvl:', classes[-1], ';sum:', sum, ';min_th:', min_threshold, ';max_th:', max_threshold)
                    if min_threshold < sum:
                        self.count_population_left -= sum
                        df_classes = pd.concat([df_classes, pd.DataFrame({'Classes':[[classes[0], classes[-1]]], 'Count': [sum], 'Pop left':[self.count_population_left]})], ignore_index=True)
                        min_threshold = sum 
                        break
            df_classes.drop(df_classes.tail(1).index, inplace=True)
        else:
            df_classes = pd.concat([df_classes, pd.DataFrame({'Classes':[[min_level, self.serie.max()]], 'Count': [count_population_left], 'Pop left':[0]})], ignore_index=True)
       
        return df_classes
    

    

def main():
    df = pd.read_csv('files/clean.csv')
    gen = GeneratorClasses(df['hours-per-week'], 5)
    print(gen.effectif)
    print(gen.generate_classes_for_levels(gen.serie.min(), gen.effectif, 1))

main()

#17-40 : 26551


# def generate_counter_items(serie):

#         """

#         """
#         counter = Counter(serie)
#         for i in sorted(counter.items()):
#             yield i

#     effectif = len(serie)
#     min_threshlod = int(0.1*effectif)
#     max_threshold = int(effectif - bins*min_threshlod)
#     counter = Counter(serie)

#     dict_all_classes = {}
#     while min_threshlod < max_threshold:
#         sum = 0
#         classes = []
#         for i in generate_counter_items(serie):
#             classes.append(i[0])
#             sum += i[1]
#             if min_threshlod < sum:
#                 dict_all_classes[str(classes)] = sum
#                 min_threshlod = sum 
#                 break
#     del dict_all_classes[str(classes)]
#     print(dict_all_classes)