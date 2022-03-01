from scipy import sparse as spsparse


def sparse2dict(X):
    """
    Function that transforms a scipy.sparse matrix into a dictionary (used in Graph-Informed layers)
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
    Function that transforms a dictionary of the type used in Graph-Informed layers into a scipy.sparse matrix
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


