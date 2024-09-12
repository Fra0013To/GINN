import tensorflow as tf
from graphinstructed.utils import sparse2dict, dict2sparse, add_rowcolkeys_selfloops
from scipy import sparse as spsparse
from tensorflow.python.trackable.data_structures import NoDependency


class DenseNonversatileGraphInstructed(tf.keras.layers.Dense):
    """
    General Graph-Instructed layer class (obtained as subclass of tf.keras.layers.Dense), dense implementation.
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
        :param highlighted_nodes: [Removed]
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

        super().__init__(units, activation, use_bias, kernel_initializer, bias_initializer,
                         kernel_regularizer, bias_regularizer,activity_regularizer,
                         kernel_constraint, bias_constraint,
                         **options)

    def get_config(self):

        config = super().get_config()

        config['adj_mat'] = self.adj_mat
        config['pool'] = self.pool_str
        config['num_filters'] = self.num_filters

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

        # ADD THE BIASS TO EACH ONE OF THE ? ELEMENTS OF THE FIRST DIMENSION; THEN, APPLY THE ACTIVATION FUNCTION
        out_tensor = self.activation(out_tensor + self.bias)

        # SQUEEZE IF F=1 OR APPLY THE POOLING OPERATION (IF pool IS NOT None)
        if out_tensor.shape[-1] == 1:
            out_tensor = tf.squeeze(out_tensor, axis=2)
        elif self.pool is not None:
            out_tensor = self.pool(out_tensor, axis=2)

        return out_tensor


class GraphInstructed(DenseNonversatileGraphInstructed):
    """
    Graph-Instructed layer class (obtained as subclass of DenseNonversatileGraphInstructed),
    sparse implementation.
    It implements the Versatile General Graph-Instructed (GI) layer, introduced in https://doi.org/10.48550/arXiv.2403.13781 by Della Santa F. as a further
    generalization of General GI layers defined in https://doi.org/10.3390/math10050786 by Berrone S., Della Santa F.,
    Mastropietro A., Pieraccini S., Vaccarino F.
    Given the sub-matrix N1-by-N2 of an adjacency matrix of a graph and the number of filters F (i.e., output features),
    the layer returns batch-array of shape (?, N2, F), for each batch of inputs of shape (?, N1, K), where K is the
    number of input features per graph node. The symbol "?" denotes the batch size.
    If num_filters=1 or option pool is not None, the output tensor has shape (?, N2). A general pooling operation that
    returns F' output features, 1 < F' < F, is not implemented yet.
    This layer can receive any batch-of-inputs tensor of shape (?, N1, K).
    """

    def __init__(self,
                 adj_mat,
                 rowkeys=None, colkeys=None,
                 selfloop_value=1.,
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
                 **options):
        """
        Initialization method.
        :param adj_mat: a dok_matrix, dok_array, or a dictionary describing the adjacency matrix (likewise the
            output of the spars2dict function)
        :param rowkeys/colkeys: indices of the rows/columns defyning adj_mat from the 'full' adjacency matrix
            (None by default; i.e., standard indexing). The list of rowkeys/colkeys is authomatically sorted in
            ascending order
        :param selfloop_value: value to be assigned to the selfloops
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
        :param options: see tf.keras.layers.Dense
        """

        if isinstance(adj_mat, spsparse.dok_matrix) or isinstance(adj_mat, spsparse.dok_array):
            if rowkeys is not None:
                rowkeys.sort()
            if colkeys is not None:
                colkeys.sort()
            adj_mat = sparse2dict(adj_mat, k1=rowkeys, k2=colkeys)
        elif isinstance(adj_mat, dict):
            adj_mat_keys = adj_mat.keys()

            assert 'shape' in adj_mat_keys, 'The dictionary adj_mat has not the key "shape"'
            assert 'keys' in adj_mat_keys, 'The dictionary adj_mat has not the key "keys"'
            assert 'values' in adj_mat_keys, 'The dictionary adj_mat has not the key "values"'
            assert 'shape' in adj_mat_keys, 'The dictionary adj_mat has not the key "shape"'
            assert 'keys_custom' in adj_mat_keys, 'The dictionary adj_mat has not the key "keys_custom"'
            assert 'rowkeys_custom' in adj_mat_keys, 'The dictionary adj_mat has not the key "rowkeys_custom"'
            assert 'colkeys_custom' in adj_mat_keys, 'The dictionary adj_mat has not the key "colkeys_custom"'
        else:
            raise ValueError('adj_mat is not a dok_matrix, dok_array, or a proper dictionary')

        adj_mat_original = adj_mat.copy()
        adj_mat_hat = add_rowcolkeys_selfloops(adj_mat, selfloop_value=selfloop_value)

        # ATTENTION, HERE WE STORE IN self.adj_mat THE MATRIX A^, NOT THE MATRIX A!
        super().__init__(
            adj_mat_hat,  # <---------------------- MATRIX A^, NOT A!!!
            num_filters=num_filters,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            pool=pool,
            **options
        )

        self.adj_mat_original = NoDependency(adj_mat_original)
        self.selfloop_value = selfloop_value

    def get_config(self):

        config = super().get_config()

        config['adj_mat_original'] = self.adj_mat_original
        config['selfloop_value'] = self.selfloop_value

        return config

    def build(self, input_shape):

        self.num_features = 1
        if len(input_shape) >= 3:
            self.num_features = input_shape[2]

        # FOR PRACTICAL USAGE IN THE CALL METHOD, WE RESHAPE THE FILTER W FROM SHAPE (N1, K, F) TO SHAPE (N1 K, 1, F)
        self.kernel = self.add_weight(name="kernel", shape=[int(self.adj_mat_original['shape'][0] * self.num_features),
                                                            1, self.num_filters
                                                            ],
                                      initializer=self.kernel_initializer)

        # BIAS OF SHAPE (1, N2, F)
        self.bias = self.add_weight(name="bias", shape=[1, self.adj_mat['shape'][1], self.num_filters], # <--------------------------- UNIQUE DIFFERENCE!
                                    initializer=self.bias_initializer)

        # print(self.adj_mat)
        adj_mat_hat = dict2sparse(self.adj_mat)  # <--------------------- MATRIX A^ WITH SELF-LOOPS
        adj_mat_hat = adj_mat_hat.todok()
        # CONVERSION OF A^ FROM SPARSE MATRIX TO TF-TENSOR
        adj_mat_hat_tf = tf.sparse.SparseTensor(list(adj_mat_hat.keys()),
                                                tf.cast(list(adj_mat_hat.values()), dtype=self.dtype),
                                                adj_mat_hat.shape
                                                )

        self.adj_mat_tf = adj_mat_hat_tf

    def call(self, input):

        self.num_features = 1
        if len(input.shape) == 3:
            self.num_features = input.shape[2]

        # CONVERSION OF A^ FROM SPARSE MATRIX TO TF-TENSOR
        adj_mat_hat_tf = self.adj_mat_tf

        # CREATE A TF-TENSOR A~ OF SHAPE (N1 K, N2), CONCATENATING K TIMES A^ ALONG ROW-AXIS
        adj_mat_hat_tiled = tf.sparse.concat(axis=0, sp_inputs=([adj_mat_hat_tf] * self.num_features))

        # RESHAPE INTPUT TENSOR, IF K > 1, FROM SHAPE (?, N1, K) TO SHAPE (?, N1 K)
        input_tot_tensor = input
        if len(input.shape) == 3:
            inputs_listed = [input[:, :, i] for i in range(self.num_features)]
            input_tot_tensor = tf.concat(inputs_listed, axis=1)

        # COMPUTE F W~^i MATRICES THAT, IF CONCATENATED ALONG axis2, ARE THE W~ TENSOR.
        # WE OBTAIN EACH W~^i MATRIX (SPARSE) AS THE ELEMENT-WISE MULTIPLCATION OF:
        # 1. THE adj_mat_hat_tiled TENSOR OF SHAPE (N1 K, N2)
        # 2. THE f-TH TENSOR OF SHAPE (N1 K, 1), GIVEN BY self.kernel[:, :, f]
        #
        # THEN
        #
        # MATRIX-MULTIPLICATION (?, N1 K) x (N1 K, N2) OF INPUTS AND W~^i. THE CONCATENATION ALONG axis2 COINCIDES WITH
        # THE OUTPUT TENSOR OF SHAPE (?, N2, F)

        out_tensor = []
        f = 0
        while f < self.num_filters:
            Wtilde_f = adj_mat_hat_tiled.__mul__(self.kernel[:, :, f])
            out_tensor.append(
                tf.expand_dims(tf.sparse.sparse_dense_matmul(input_tot_tensor, Wtilde_f), axis=2)
            )
            f += 1

        out_tensor = tf.concat(out_tensor, axis=2)

        # ADD THE BIAS TO EACH ONE OF THE ? ELEMENTS OF THE FIRST DIMENSION; THEN, APPLY THE ACTIVATION FUNCTION
        out_tensor = self.activation(out_tensor + self.bias)

        # SQUEEZE IF F=1 OR APPLY THE POOLING OPERATION (IF pool IS NOT None)
        if out_tensor.shape[-1] == 1:
            out_tensor = tf.squeeze(out_tensor, axis=2)
        elif self.pool is not None:
            out_tensor = self.pool(out_tensor, axis=2)

        return out_tensor









