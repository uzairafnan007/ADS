import numpy as np
import seaborn as sns
import sklearn
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
import csv
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.linear_model import LinearRegression
mtcars = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data
df = pd.DataFrame(mtcars)
df_=df.copy()
df.head()
df.info()
df.describe()
# df.isnull().sum()
# df.duplicated().sum()
# df.drop_duplicates(inplace=True)
with open('output.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['head', 'tail', 'sample', 'columns', 'index', 'dtypes', 'mean', 'median', 'mode', 'std', 'var', 'skew', 'kurtosis'])
    csvwriter.writerow([df.head(), df.tail(), df.sample(5), df.columns, df.index, df.dtypes, df.mean(), df.median(), df.mode(), df.std(), df.var(), df.skew(), df.kurtosis()])
    print(df.head())
    print(df.tail())
    print(df.sample(5))
    print(df.columns)
    print(df.index)
    print(df.dtypes)
    print(df.mean())
    print(df.median())
    print(df.mode())
    print(df.std())
    print(df.var())
    print(df.skew())
    print(df.kurtosis())
sns.kdeplot(df.mpg,fill=True,color="blue")
plt.show()
sns.kdeplot(df.disp,fill=True,color="red")
plt.show()
sns.kdeplot(data=df, x='mpg', y='disp',color='r', fill=True, cmap="Greens")
plt.show()
sns.kdeplot(data=df, x='mpg', y='hp',color='r', fill=True, cmap="viridis")
plt.show()

