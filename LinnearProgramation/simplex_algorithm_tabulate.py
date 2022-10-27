import numpy as np

def simplex_algorithm_tabulate(table):
    zk_ck = max(table[0][2])
    k = np.where(table[0][2] == zk_ck)
    while zk_ck > 0:
        XBRHS = table[1][3]
        XBXN = table[1][2]
        Xks = list()
        for i in range(len(XBRHS)):
            if XBXN[k][i] > 0:
                Xks.append(XBRHS[i] / XBXN[k][i])
        Xk = min(Xks)
        simplex_algorithm_tabulate(table)
    pass