import tensorflow as tf
import numpy as np
from graphinformed.layers import GraphInformed
from graphinformed.utils import sparse2dict, dict2sparse, add_rowcolkeys_selfloops
from scipy import sparse as spsparse

random_seed = 42

np.random.seed(random_seed)

# Number of graph nodes
N = 5

# Number of training samples
T = 50

# Random adjacency matrix
A = np.random.rand(N, N)
A = 0.5 * (A + A.T)
A[[i for i in range(N)], [i for i in range(N)]] = 0

# Convert the matrix A into a sparse matrix and then into a dictionary
Asparse = spsparse.dok_matrix(A)
# CONVERSION INTO A DICTIONARY (Necessary for dense, old, implementation)
# Adict = sparse2dict(Asparse)


# -------- GINN creation -----------

print('@@@@@@@@@@ START OF GINN MODEL CREATION @@@@@@@@')

# Number of filters
F = 10

# V2 nodes (target)
V2_list = [1, 3]
subAsparse = spsparse.dok_matrix(A[:, V2_list])
# CONVERSION INTO A DICTIONARY (not necessary)
# subAdict = sparse2dict(subAsparse, k2=V2_list)

# Creation of a fake training
X = np.random.rand(T, N)
Y = np.random.rand(T, len(V2_list))

I = tf.keras.layers.Input(N)
G1 = GraphInformed(adj_mat=Asparse,
                   num_filters=F
                   )(I)
G2 = GraphInformed(adj_mat=Asparse,
                   num_filters=F
                   )(G1)
G3 = GraphInformed(adj_mat=subAsparse,
                   colkeys=V2_list,
                   activation='linear',
                   num_filters=F,
                   pool='reduce_max'
                   )(G2)

model = tf.keras.models.Model(inputs=I, outputs=G3)

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mae']
              )

print('@@@@@@@@@@ GINN MODEL CREATION COMPLETED @@@@@@@@')

# -------- GINN (fake) training -----------


print('@@@@@@@@@@ START OF GINN TRAINING @@@@@@@@')


model.fit(X, Y,
          epochs=5,
          batch_size=T // 5,
          verbose=True
          )

print('@@@@@@@@@@ GINN TRAINING COMPLETED @@@@@@@@')


print('@@@@@@@@@@ EXAMPLE OF GINN PREDICITONS @@@@@@@@')
print(model.predict(X[:T // 5, :]))






