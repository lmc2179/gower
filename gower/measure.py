import numpy as np

# TODO: Support for list of lists, panda dataframes'
# TODO: Type validation

R = 'R'
B = 'B'
C = 'C'

def _tile_column(col):
    r = np.vstack([col for _ in range(len(col))])
    return r.T, r

def similarity(X, types):
    s_matrix = np.zeros((len(X), len(X)))
    d_matrix = np.zeros((len(X), len(X)))
    for c, t in enumerate(types):
        col_vector = X[:,c]
        col_matrix, row_matrix = _tile_column(col_vector)
        if t == R:
            r_k = max(col_vector) - min(col_vector)
            s_matrix = s_matrix + (1.0 - (1.*np.abs(col_matrix - row_matrix) / r_k))
            d_matrix = d_matrix + np.ones(d_matrix.shape)
        elif t == B:
            s_matrix = s_matrix + ((col_matrix + row_matrix) == 2)
            d_matrix = d_matrix + (1. - ((col_matrix + row_matrix) == 0))
        elif t == C:
            s_matrix = s_matrix + (col_matrix == row_matrix)
            d_matrix = d_matrix + np.ones(d_matrix.shape)
        else: 
            raise Exception('unrecognized type {0}'.format(t))
    return s_matrix / d_matrix

def distance(X, types):
    return 1.0 - similarity(X, types)
