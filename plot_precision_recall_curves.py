import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette('muted')

df = pd.read_csv('results/chi_2018/precision_recall_dataframe.csv') 
print(df.head())

g = sns.FacetGrid(
    data=df,
    row='algo_name', col='C_val',
    hue='max_features',
)
g = (g.map(plt.plot, 'recall', 'precision').add_legend())

plt.show()