from scipy import sparse as spsparse


def sparse2dict(X, k1=None, k2=None):
    """
    Function that transforms a scipy.sparse matrix into a dictionary (used in Graph-Instructed layers)
    :param X: scipy.sparse matrix
    :param k1, k2: list of indexes identifying submatrix (custom order!)
    :return: dictionary characterizing (a sub-matrix of) an adjacency matrix described by X, k1, and k2.
    """
    Xdok = X.todok()
    dict_X = {}
    keys = list(Xdok.keys())
    dict_X['keys'] = keys

    if k1 is not None and k2 is not None:
        dict_X['keys_custom'] = [(k1[t[0]], k2[t[1]]) for t in keys]
        dict_X['rowkeys_custom'] = k1
        dict_X['colkeys_custom'] = k2
    elif k1 is not None:
        dict_X['keys_custom'] = [(k1[t[0]], t[1]) for t in keys]
        dict_X['rowkeys_custom'] = k1
        dict_X['colkeys_custom'] = None  # THEORETICALLY, THEY ARE: list(range(Xdok.shape[1]))
    elif k2 is not None:
        dict_X['keys_custom'] = [(t[0], k2[t[1]]) for t in keys]
        dict_X['rowkeys_custom'] = None  # THEORETICALLY, THEY ARE: list(range(Xdok.shape[0]))
        dict_X['colkeys_custom'] = k2
    else:
        dict_X['keys_custom'] = None  # THEORETICALLY, THEY ARE: keys.copy()
        dict_X['rowkeys_custom'] = None  # THEORETICALLY, THEY ARE: list(range(Xdok.shape[0]))
        dict_X['colkeys_custom'] = None  # THEORETICALLY, THEY ARE: list(range(Xdok.shape[1]))

    dict_X['values'] = list(Xdok.values())
    dict_X['shape'] = Xdok.shape

    return dict_X


def add_rowcolkeys_selfloops(dict_X, selfloop_value=1.):
    """
    Given a dictionary of the type returned by sparse2dict (assuming NO SELF-LOOPS), this function add self-loops.
    This operation is not trivial in case of X sub-matrix of the adjacency matrix of the "full" graph.
    :param dict_X: dictionary of the type returned by sparse2dict
    :param selfloop_value: value to be assigned to the selfloops
    :return: same dictionary, updated with proper self-loops
    """
    if dict_X['keys_custom'] is None:
        rowcolkeys_custom = list(set(range(dict_X['shape'][0])).intersection(set(range(dict_X['shape'][1]))))

        for rc in rowcolkeys_custom:
            dict_X['keys'].append((rc, rc))
            dict_X['values'].append(selfloop_value)

    else:
        if dict_X['rowkeys_custom'] is None:
            rowkeys_custom = list(range(dict_X['shape'][0]))
        else:
            rowkeys_custom = dict_X['rowkeys_custom']

        rowkeys_pairs_dict = dict(zip(rowkeys_custom, list(range(dict_X['shape'][0]))))

        if dict_X['colkeys_custom'] is None:
            colkeys_custom = list(range(dict_X['shape'][1]))
        else:
            colkeys_custom = dict_X['colkeys_custom']

        colkeys_pairs_dict = dict(zip(colkeys_custom, list(range(dict_X['shape'][1]))))

        rowcolkeys_custom = list(set(rowkeys_custom).intersection(set(colkeys_custom)))

        for rc in rowcolkeys_custom:
            dict_X['keys_custom'].append((rc, rc))
            dict_X['keys'].append((rowkeys_pairs_dict[rc], colkeys_pairs_dict[rc]))
            dict_X['values'].append(selfloop_value)

    return dict_X


def dict2sparse(dict_X, sparse_type='dok_matrix'):
    """
    Function that transforms a dictionary of the type returned by sparse2dict into a scipy.sparse matrix
    :param dict_X: dictionary of the type returned by sparse2dict
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