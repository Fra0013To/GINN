import tensorflow as tf
from scipy import sparse as spsparse


def sparse2dict(X):
    """
    Function that transforms a scipy.sparse matrix into a dictionary (used in Graph-Instructed layers)
    :param X: scipy.sparse matrix
    :return: dictionary that contains attributes of X.tocoo() object, converted to lists
    """
    Xdok = X.todok()
    dict_X = {}
    dict_X['keys'] = list(Xdok.keys())
    dict_X['values'] = list(Xdok.values())
    dict_X['shape'] = Xdok.shape

    return dict_X


def dict2sparse(dict_X, sparse_type='dok_matrix'):
    """
    Function that transforms a dictionary of the type used in Graph-Instructed layers into a scipy.sparse matrix
    :param dict_X: dictionary that contains attributes of X.tocoo() object (X scipy.sparse), converted to lists
    :param sparse_type: type of scipy.sparse for X (default 'coo_matrix'). All the possibilities are:
        'bsr_matrix': Block sparse row matrix)
        'coo_matrix': sparse matrix in COOrdinate format)
        'csc_matrix': Compressed Sparse Column matrix)
        'csr_matrix': Compressed Sparse Row matrix
        'dia_matrix': Sparse matrix with DIAgonal storage
        'dok_matrix': Dictionary Of Keys based sparse matrix.
        'lil_matrix': Row-based linked list sparse matrix
    :return: scipy.sparse of the chosen type
    """
    X = spsparse.dok_matrix(tuple(dict_X['shape']))
    for i in range(len(dict_X['values'])):
        X[dict_X['keys'][i][0], dict_X['keys'][i][1]] = dict_X['values'][i]

    if sparse_type == 'bsr_matrix':
        X = X.tobsr()
    elif sparse_type == 'coo_matrix':
        X = X.tocoo()
    elif sparse_type == 'csc_matrix':
        X = X.tocsc()
    elif sparse_type == 'csr_matrix':
        X = X.tocsr()
    elif sparse_type == 'dia_matrix':
        X = X.todia()
    elif sparse_type == 'lil_matrix':
        X = X.tolil()

    return X


class OldDenseGraphInstructed(tf.keras.layers.Dense):
    """
    Graph-Instructed layer class (obtained as subclass of tf.keras.layers.Dense).
    It implements the Graph-Instructed layer introduced in https://doi.org/10.3390/math10050786
    by Berrone S., Della Santa F., Mastropietro A., Pieraccini S., Vaccarino F..
    Given the adjacency matrix of shape (N, N) of a graph and the number of filters F (i.e., output features) the layer
    returns batch-array of shape (?, N, F), for each batch of inputs of shape (?, N, K), where K is the number of
    input features per graph node.
    The symbol "?" denotes the batch size.
    If num_filters=1 or option pool is not None, the output tensor has shape (?, N). A general pooling operation that
    returns F' output features, 1 < F' < F, is not implemented yet.
    This layer can receive any batch-of-inputs tensor of shape (?, N, K).
    If highlighted_nodes is a list of m indexes of graph nodes, the tensor returned by the layer has shape
    (?, m, F) (or (?, m)), where the output features are the ones related to the selected m nodes of the
    graph (i.e., performs the "mask" operation illustrated in the paper).
    """
    def __init__(self,
                 adj_mat,
                 num_filters=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 pool=None,
                 highlighted_nodes=None,
                 **options):
        """
        Initialization method.
        :param adj_mat: a dictionary describing the adjacency matrix, likewise the output of the spars2dict function
        :param num_filters: the number F of output features of the layer.
        :param activation: see tf.keras.layers.Dense
        :param use_bias: see tf.keras.layers.Dense
        :param kernel_initializer: see tf.keras.layers.Dense
        :param bias_initializer: see tf.keras.layers.Dense
        :param kernel_regularizer: see tf.keras.layers.Dense
        :param bias_regularizer: see tf.keras.layers.Dense
        :param activity_regularizer: see tf.keras.layers.Dense
        :param kernel_constraint: see tf.keras.layers.Dense
        :param bias_constraint: see tf.keras.layers.Dense
        :param pool:  None or a string denoting a tf-reducing function (e.g.: 'reduce_mean', 'reduce_max', ...).
            Default is None.
        :param highlighted_nodes: None or list of indexes of the graph nodes on which we focus. Default is None.
        :param options: see tf.keras.layers.Dense
        """

        if isinstance(adj_mat, dict):
            self.adj_mat = adj_mat
        else:
            raise ValueError('adj_mat is not a dictionary')

        adj_mat_keys = adj_mat.keys()

        assert 'shape' in adj_mat_keys, 'The dictionary adj_mat has not the key "shape"'
        assert 'keys' in adj_mat_keys, 'The dictionary adj_mat has not the key "keys"'
        assert 'values' in adj_mat_keys, 'The dictionary adj_mat has not the key "values"'

        units = adj_mat['shape'][0]

        self.num_filters = num_filters
        self.num_features = 1

        if pool is None:
            self.pool = None
            self.pool_str = None
        else:
            self.pool = getattr(tf, pool)
            self.pool_str = pool

        self.highlighted_nodes = highlighted_nodes

        super().__init__(units, activation, use_bias, kernel_initializer, bias_initializer,
                         kernel_regularizer, bias_regularizer,activity_regularizer,
                         kernel_constraint, bias_constraint,
                         **options)

    def get_config(self):

        config = super().get_config()

        config['adj_mat'] = self.adj_mat
        config['pool'] = self.pool_str
        config['num_filters'] = self.num_filters
        config['highlighted_nodes'] = self.highlighted_nodes

        del config['units']

        return config

    def build(self, input_shape):

        self.num_features = 1
        if len(input_shape) >= 3:
            self.num_features = input_shape[2]

        # FOR PRACTICAL USAGE IN THE CALL METHOD, WE RESHAPE THE FILTER W FROM SHAPE (N, K, F) TO SHAPE (NK, 1, F)
        self.kernel = self.add_weight(name="kernel", shape=[int(self.units * self.num_features), 1, self.num_filters],
                                      initializer=self.kernel_initializer)

        # BIAS OF SHAPE (1, N, F)
        self.bias = self.add_weight(name="bias", shape=[1, self.units, self.num_filters],
                                    initializer=self.bias_initializer)

    def call(self, input):

        self.num_features = 1
        if len(input.shape) == 3:
            self.num_features = input.shape[2]

        adj_mat_loaded = dict2sparse(self.adj_mat)
        adj_mat_hat = adj_mat_loaded + spsparse.identity(self.units)

        # CONVERSION OF A^ FROM SPARSE MATRIX TO TF-TENSOR
        adj_mat_hat_tf = tf.cast(adj_mat_hat.toarray(), dtype=input.dtype)

        # CREATE A TF-TENSOR A~ OF SHAPE (NK, N, 1), CONCATENATING K TIMES A^ ALONG ROW-AXIS
        adj_mat_hat_tiled_expanded = tf.expand_dims(tf.concat([adj_mat_hat_tf] * self.num_features, axis=0),
                                                    axis=2
                                                    )

        # COMPUTE THE W~ TENSOR MULTIPLYING ELEMENT-WISE:
        # 1. A TF-TENSOR OF SHAPE (NK, N, F), OBTAINED CONCATENATING F TIMES THE TENSOR A~ ALONG THE 3rd AXIS
        # 2. THE self.kernel TENSOR OF SHAPE (NK, 1, F)
        Wtilde = tf.tile(adj_mat_hat_tiled_expanded, [1, 1, self.num_filters]) * self.kernel

        # RESHAPE INTPUT TENSOR, IF K > 1, FROM SHAPE (?, N, K) TO SHAPE (?, NK)
        input_tot_tensor = input
        if len(input.shape) == 3:
            inputs_listed = [input[:, :, i] for i in range(self.num_features)]
            input_tot_tensor = tf.concat(inputs_listed, axis=1)

        # SIMPLE MULTIPLICATION (?, NK) x (NK, N, F), OBTAINING THE OUTPUT TENSOR OF SHAPE (?, N, F)
        out_tensor = tf.tensordot(input_tot_tensor, Wtilde, [[1], [0]])

        # ADD THE BIAS TO EACH ONE OF THE ? ELEMENTS OF THE FIRST DIMENSION; THEN, APPLY THE ACTIVATION FUNCTION
        out_tensor = self.activation(out_tensor + self.bias)

        # FOCUS ONLY ON THE SELECTED NODE (self.highlighted_nodes)
        if self.highlighted_nodes is not None:
            out_tensor = tf.gather(out_tensor, tf.constant(self.highlighted_nodes), axis=1)

        # SQUEEZE IF F=1 OR APPLY THE POOLING OPERATION (IF pool IS NOT None)
        if out_tensor.shape[-1] == 1:
            out_tensor = tf.squeeze(out_tensor, axis=2)
        elif self.pool is not None:
            out_tensor = self.pool(out_tensor, axis=2)

        return out_tensor


class EdgeWiseGraphInstructed(OldDenseGraphInstructed):
    """
    Graph-Instructed layer class (obtained as subclass of tf.keras.layers.Dense).
    It implements the Edge-Wise Graph-Instructed layer introduced in TODO: ADD PAPER REFERENCE
    by Della Santa F., Mastropietro A., Pieraccini S., Vaccarino F..
    Given the adjacency matrix of shape (N, N) of a graph and the number of filters F (i.e., output features) the layer
    returns batch-array of shape (?, N, F), for each batch of inputs of shape (?, N, K), where K is the number of
    input features per graph node.
    The symbol "?" denotes the batch size.
    If num_filters=1 or option pool is not None, the output tensor has shape (?, N). A general pooling operation that
    returns F' output features, 1 < F' < F, is not implemented yet.
    This layer can receive any batch-of-inputs tensor of shape (?, N, K).
    If highlighted_nodes is a list of m indexes of graph nodes, the tensor returned by the layer has shape
    (?, m, F) (or (?, m)), where the output features are the ones related to the selected m nodes of the
    graph (i.e., performs the "mask" operation illustrated in the paper).
    """

    def build(self, input_shape):

        self.num_features = 1
        if len(input_shape) >= 3:
            self.num_features = input_shape[2]

        # FOR PRACTICAL USAGE IN THE CALL METHOD, WE RESHAPE THE STRAIGHT-FILTER W FROM SHAPE (N, K, F)
        # TO SHAPE (NK, 1, F)
        self.kernel_straight = self.add_weight(name="kernel_straight",
                                               shape=[int(self.units * self.num_features), 1, self.num_filters],
                                               initializer=self.kernel_initializer
                                               )

        # FOR PRACTICAL USAGE IN THE CALL METHOD, WE KEEP THE REVERSE-FILTER Wrev WITH A SHAPE OF (N, K, F)
        self.kernel_rev = self.add_weight(name="kernel_rev", shape=[self.units, self.num_features, self.num_filters],
                                          initializer=self.kernel_initializer)

        # BIAS OF SHAPE (1, N, F)
        self.bias = self.add_weight(name="bias", shape=[1, self.units, self.num_filters],
                                    initializer=self.bias_initializer)

    def call(self, input):

        self.num_features = 1
        if len(input.shape) == 3:
            self.num_features = input.shape[2]

        adj_mat_loaded = dict2sparse(self.adj_mat)
        adj_mat_hat = adj_mat_loaded + spsparse.identity(self.units)

        # CONVERSION OF A^ FROM SPARSE MATRIX TO TF-TENSOR
        adj_mat_hat_tf = tf.cast(adj_mat_hat.toarray(), dtype=input.dtype)

        # CREATE A TF-TENSOR A^~ OF SHAPE (N, N, F), CONCATENATING F TIMES THE TENSOR A^ ALONG THE 3rd AXIS
        adj_mat_hat_tiled_expanded = tf.expand_dims(adj_mat_hat_tf, axis=2)
        adj_mat_hat_tiled_expanded = tf.concat([adj_mat_hat_tiled_expanded] * self.num_filters, axis=2)

        # CREATE A LIST OF TENSORS A^~ * Wrev(:, k, :), FOR k = 1, ..., K
        adj_mat_hat_tiled_expanded_list = [adj_mat_hat_tiled_expanded * self.kernel_rev[:, k:k + 1, :]
                                           for k in range(self.num_features)
                                           ]
        # COMPUTE THE W~ TENSOR OF SHAPE (NK, N, F) MULTIPLYING ELEMENT-WISE THE FOLLOWING TENSORS:
        # 1. THE CONCATENATION OF THE TENSORS IN THE PREVIOUS LIST ALONG THE 1st AXIS
        # 2. THE self.kernel_straight TENSOR OF SHAPE (NK, 1, F)
        Wtilde = tf.concat(adj_mat_hat_tiled_expanded_list, axis=0)
        Wtilde = Wtilde * self.kernel_straight

        # RESHAPE INTPUT TENSOR, IF K > 1, FROM SHAPE (?, N, K) TO SHAPE (?, NK)
        input_tot_tensor = input
        if len(input.shape) == 3:
            inputs_listed = [input[:, :, i] for i in range(self.num_features)]
            input_tot_tensor = tf.concat(inputs_listed, axis=1)

        # SIMPLE MULTIPLICATION (?, NK) x (NK, N, F), OBTAINING THE OUTPUT TENSOR OF SHAPE (?, N, F)
        out_tensor = tf.tensordot(input_tot_tensor, Wtilde, [[1], [0]])

        # ADD THE BIAS TO EACH ONE OF THE ? ELEMENTS OF THE FIRST DIMENSION; THEN, APPLY THE ACTIVATION FUNCTION
        out_tensor = self.activation(out_tensor + self.bias)

        # FOCUS ONLY ON THE SELECTED NODE (self.highlighted_nodes)
        if self.highlighted_nodes is not None:
            out_tensor = tf.gather(out_tensor, tf.constant(self.highlighted_nodes), axis=1)

        # SQUEEZE IF F=1 OR APPLY THE POOLING OPERATION (IF pool IS NOT None)
        if out_tensor.shape[-1] == 1:
            out_tensor = tf.squeeze(out_tensor, axis=2)
        elif self.pool is not None:
            out_tensor = self.pool(out_tensor, axis=2)

        return out_tensor

