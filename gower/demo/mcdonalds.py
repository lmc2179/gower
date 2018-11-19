from gower import measure
import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter

df = pd.read_csv('menu.csv') # https://www.kaggle.com/mcdonalds/nutrition-facts/version/1

df = df[df['Serving Size'].apply(lambda s: ' g)' in s)]

df['Grams per serving'] = df['Serving Size'].apply(lambda s: float(s[s.index('(')+1:s.index(' g')]))

numeric_cols = ['Sugars', 'Protein', 'Calories', 'Total Fat', 'Cholesterol', 'Sodium', 'Carbohydrates', 'Dietary Fiber']

X = df[numeric_cols + ['Grams per serving', 'Category']]
for c in numeric_cols:
    X[c] = X[c] / X['Grams per serving']
D = measure.distance(X.values, ['R' for _ in range(len(numeric_cols))] + ['R', 'C'])

dim = MDS(dissimilarity='precomputed')
X_transformed = dim.fit_transform(D)

plt.scatter(X_transformed[:,0], X_transformed[:,1])

transformed_df = pd.DataFrame({'name':df['Item'], 'x':X_transformed[:,0], 'y':X_transformed[:,1]})
print(transformed_df)

plt.show()

# NN in the transformed space

nbrs = NearestNeighbors(n_neighbors=3).fit(X_transformed)
_, indices = nbrs.kneighbors(X_transformed)
points = indices[:,0]
closest_neighbor_1 = indices[:,1]
closest_neighbor_2 = indices[:,2]
for p, n, n2 in zip(points, closest_neighbor_1, closest_neighbor_2):
    print(df['Item'].iloc[p], ' is closest to ' , df['Item'].iloc[n], ', ', df['Item'].iloc[n2])
