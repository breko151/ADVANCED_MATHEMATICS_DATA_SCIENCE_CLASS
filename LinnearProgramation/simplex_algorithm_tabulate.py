# Author: Suárez Pérez Juan Pablo 
# Date: 08/11/2022

import numpy as np

def simplex_tableau(tableau, basic_variables, non_basic_variables, itera = 1):
    
    # Printing our initial parameters for this iteration
    print("\n\n\033[1mIteration ", itera, "\033[0m")
    print("[\033[94m x_B \033[0m, \033[91m x_N \033[0m] = [ \033[94m"
          + ', '.join(basic_variables) + "\033[0m, \033[91m" + ', '.join(non_basic_variables) + "\033[0m ]\n")
    
    # Extracting elements from the tableau of the current iteration
    m = np.shape(tableau)[0] - 1
    n = np.shape(tableau)[1] - 2
    one_array = np.array([tableau[0, 0]])
    zero_array_h = np.ravel(tableau[0, 1:m + 1])
    zero_array_v = tableau[1:m + 1, 0].reshape(1, m)
    identity_matrix = tableau[1:, 1:m + 1]
    ZXN = np.ravel(tableau[0, m + 1:n + 1])
    XBXN = tableau[1:, m + 1:n + 1]
    ZRHS = np.array([tableau[0, n + 1]])
    XBRHS = (tableau[1:, n + 1]).reshape(1, m)
    b_ = np.concatenate((np.ravel(XBRHS), np.zeros(n - m)))
    
    print("\nEntry tableau")
    print(tableau)
    
    z_minus_c_max = max(ZXN)
    k = np.argmax(ZXN)
    
    # Validating if we stop or not the optimization
    if z_minus_c_max <= 0:   # We stop
        
        # Printing the final results
        print("\n\n\033[1mOptimality reached\033[0m")
        print("\nThe optimal BFS is")
        results_ls = [str(round(i, 4)) for i in b_]
        print("\033[92m[ " + ', '.join(basic_variables + non_basic_variables) + " ] = [ " + ', '.join(results_ls) + " ]\033[0m")
        perf_z = ZRHS[0]
        print("\nWith performance z =", perf_z)
        
    else:   # We continue
        
        # Calculating and printing the y_ki's
        y_k = np.ravel(XBXN[:, k])
        
        print("\nk =", k + 1, "-> column no.", k + 1, "of x_N part (", non_basic_variables[k], ")")
        print("y_k")
        print(y_k)
        
        # Analyzing if the optimal BFS is or not boundable by the condition y_k > 0
        flag = True
        aux_counter = 0
        for y_i in y_k:
            if y_i > 0:
                aux_counter = aux_counter + 1      
        if aux_counter == 0:
            flag = False

        if flag:   # Boundable
            
            # Calculating and printing x_k by the minimum quotient (current BFS divided by the y_k > 0) with its index r
            quot_ls = list()
            index_r_ls = list()
            for i in range(m):
                if y_k[i] > 0:
                    quot_ls.append(b_[i] / y_k[i])
                    index_r_ls.append(i)
            quot_arr = np.array(quot_ls)
            index_r_arr = np.array(index_r_ls)
            r = index_r_arr[np.argmin(quot_arr)]
            x_Br = min(quot_arr)
            print("\nr =", r + 1, "-> column no.", r + 1, "of x_B part (", basic_variables[r], ")")
            print("x_Br")
            print(x_Br)
            
            # Pivoting
            pivot = y_k[r]
            print("\npivot =", pivot)
            
            tableau[r + 1, :] = tableau[r + 1, :] / pivot
            for i in range(m + 1):
                if i != r + 1:
                    tableau[i, :] = tableau[i, :] - tableau[i, m + k + 1] * tableau[r + 1, :]
            
            print("\nPivoted tableau")
            print(tableau)
            # Exchanging the indexes
            print("\n\033[94m", non_basic_variables[k], "enters\033[0m and \033[91m", basic_variables[r], "leaves\033[0m the basis")
            aux_vars = basic_variables[r]
            basic_variables[r] = non_basic_variables[k]
            non_basic_variables[k] = aux_vars
            
            # Recursive call
            simplex_tableau(tableau, basic_variables, non_basic_variables, itera + 1)

        else:   # Not boundable
            
            print("\n\n\033[1mOptimization process stopped :(\033[0m")
            print("\nThe optimal BFS is not boundable")



def generate_table(A, b, c, variables, basic_indexes, non_basic_indexes, m, n):
    basic_variables = variables[m:n]
    non_basic_variables = variables[:m]

    B = A[:m, basic_indexes]
    N = A[:m, non_basic_indexes]
    c_b = c[basic_indexes]
    c_n = c[non_basic_indexes]
    b_ = np.linalg.inv(B) @ b

    one_array = np.ones(1, dtype=int)
    zero_array_h = np.ravel(np.zeros(m, dtype=int))
    zero_array_v = zero_array_h.reshape(1, m)
    identity_matrix = np.eye(m)
    ZXN = np.ravel((c_b @ np.linalg.inv(B) @ N) - c_n)
    XBXN = np.linalg.inv(B) @ N
    ZRHS = np.full((1), c_b @ np.ravel(b_))
    XBRHS = b_.reshape(1, m)

    tableau = np.vstack((np.concatenate((one_array, zero_array_h, ZXN, ZRHS), axis = 0).reshape(1, n + 2),
                np.concatenate((zero_array_v.T, identity_matrix, XBXN, XBRHS.T), axis = 1)))
    
    return [tableau, basic_variables, non_basic_variables]