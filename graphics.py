import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('revenus.csv')

def boxplot(df):
    plt.figure()
    df.plot(kind='box', subplots=True, figsize=(22,6))
    plt.show()

def pieplot(df):
    plt.figure()
    df.select_dtypes(include='object').plot(kind='pie',subplots=True, figsize=(22,6))
    plt.show()

def main():
    df = pd.read_csv('revenus.csv')
    #boxplot(df)
    pieplot(df)