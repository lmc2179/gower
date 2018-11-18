import gower
import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter

df = pd.read_csv('beers.csv')

df = df[df['abv'].notnull()]
df = df[df['ibu'].notnull()]
df['style'] = df['style'].apply(str)
df['abv'] = df['abv'].apply(float)
df['ibu'] = df['ibu'].apply(float)

print(df.dtypes)

#X = df[['abv', 'ibu', 'brewery_id', 'style']]
#D = gower.distance(X.values, ['R', 'R', 'C', 'C'])

X = df[['abv', 'ibu', 'brewery_id']]
D = gower.distance(X.values, ['R', 'R', 'C'])

dim = MDS(dissimilarity='precomputed')
X_transformed = dim.fit_transform(D)

plt.scatter(X_transformed[:,0], X_transformed[:,1])

ipa = df['style'].apply(lambda s: 'ipa' in s.lower())

print(len(ipa))
print(sum(ipa))
print(df[ipa])
print(X_transformed[ipa][:,0])
plt.scatter(X_transformed[ipa][:,0], X_transformed[ipa][:,1], color='red', marker='o')

plt.show()
