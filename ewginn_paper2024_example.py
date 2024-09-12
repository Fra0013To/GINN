import tensorflow as tf
import numpy as np
from graphinstructed.layers_ewginn_paper2024 import OldDenseGraphInstructed, EdgeWiseGraphInstructed, sparse2dict
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
Adict = sparse2dict(Asparse)


# -------- GINN creation -----------

print('@@@@@@@@@@ START OF GINN MODEL CREATION @@@@@@@@')

# Number of filters
F = 10

# Highlighted nodes (mask operation)
hn_list = [1, 3]

# Creation of a fake training
X = np.random.rand(T, N)
Y = np.random.rand(T, len(hn_list))

I = tf.keras.layers.Input(N)
G1 = OldDenseGraphInstructed(adj_mat=Adict,
                             num_filters=F
                             )(I)
G2 = EdgeWiseGraphInstructed(adj_mat=Adict,
                             num_filters=F
                             )(G1)
G3 = EdgeWiseGraphInstructed(adj_mat=Adict,
                             num_filters=F
                             )(G2)
Gout = OldDenseGraphInstructed(adj_mat=Adict,
                               activation='linear',
                               num_filters=F,
                               highlighted_nodes=hn_list,
                               pool='reduce_max'
                               )(G3)

model = tf.keras.models.Model(inputs=I, outputs=Gout)

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





