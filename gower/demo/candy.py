from gower import measure
import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter



NAME = 'competitorname'
CHOCOLATE = 'chocolate'
FRUITY = 'fruity'
CARAMEL = 'caramel'
NUTTY = 'peanutyalmondy'
NOUGAT = 'nougat'
WAFER = 'crispedricewafer'
HARD = 'hard'
BAR = 'bar'
PLURIBUS = 'pluribus'
SUGAR = 'sugarpercent'
PRICE = 'pricepercent'


df = pd.read_csv('candy-data.csv') # https://www.kaggle.com/fivethirtyeight/fivethirtyeight-candy-power-ranking-dataset

numeric_cols = [SUGAR, PRICE]
categorical_cols = [CHOCOLATE, FRUITY, CARAMEL, NUTTY, NOUGAT, WAFER, HARD, BAR, PLURIBUS]

X = df[numeric_cols + categorical_cols]
D = measure.distance(X.values, ['R' for _ in range(len(numeric_cols))] + ['C' for _ in range(len(categorical_cols))])

dim = MDS(dissimilarity='precomputed')
X_transformed = dim.fit_transform(D)

plt.scatter(X_transformed[:,0], X_transformed[:,1])

transformed_df = pd.DataFrame({'name':df[NAME], 'x':X_transformed[:,0], 'y':X_transformed[:,1]})
print(transformed_df)

plt.show()

# NN in the transformed space

nbrs = NearestNeighbors(n_neighbors=3).fit(X_transformed)
_, indices = nbrs.kneighbors(X_transformed)
points = indices[:,0]
closest_neighbor_1 = indices[:,1]
closest_neighbor_2 = indices[:,2]
for p, n, n2 in zip(points, closest_neighbor_1, closest_neighbor_2):
    print(df[NAME].iloc[p], ' is closest to ' , df[NAME].iloc[n], ', ', df[NAME].iloc[n2])
