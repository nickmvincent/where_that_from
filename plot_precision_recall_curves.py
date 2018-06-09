import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette('muted')


df = pd.read_csv('results/psa_research/precision_recall_dataframe.csv') 
# g = sns.FacetGrid(data=df, col='name', hue='max_features', col_wrap=4)
# g = g.map(sns.tsplot, data=df, time='recall', value='precision', color="r")

print(df.head())

g = sns.FacetGrid(
    data=df,
    row='algo_name', col='C_val',
    hue='max_features',
)
g = (g.map(plt.plot, 'recall', 'precision').add_legend())

plt.show()