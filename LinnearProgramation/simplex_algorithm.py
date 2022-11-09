# Author: Suárez Pérez Juan Pablo 
# Date: 08/11/2022

# Import Libraries
import numpy as np

def simplex(A, b, c, variables, itera = 1):
    
    # Printing our initial parameters for this iteration
    print("\n\n\033[1mIteration ", itera, "\033[0m")
    print("[\033[94m x_B \033[0m, \033[91m x_N \033[0m] = [ \033[94m"
          + ', '.join(variables[:len(b)]) + "\033[0m, \033[91m" + ', '.join(variables[len(b):]) + "\033[0m ]\n")
    print("Matrix A")
    print(A)
    print("\nVector b")
    print(b)
    print("\nVector c")
    print(c)
    
    # Get a basic matrix B and a not basic matrix N
    m, n = np.shape(A)
    B = A[:m, :m]
    N = A[:m, m:]
    
    # Get B inverse
    B_inv = np.linalg.inv(B)
    # Get xb
    x_b = np.array(B_inv @ b)
    x_b = x_b[0, :]
    # Get xn
    x_n = np.zeros(n - m)
    # Get x
    x = np.concatenate((x_b, x_n), axis = 0)
    # Print Results
    print("\nBasis")
    print(B)
    print("\nBasis Inverse")
    print(B_inv)
    print("\nBFS")
    print(x)
    
    # Calculating the z's minus c's and finding the maximum one with its index k
    j_index = [i for i in range(m, n)]
    c_b = c[:m]
    zs = np.zeros(len(j_index))
    z_minus_c = np.zeros(len(j_index))
    for i in range(len(j_index)):
        a_j = A[:,j_index[i]]
        zs[i] = c_b@B_inv@a_j
        z_minus_c[i] = zs[i] - c[j_index[i]]
        
    z_minus_c_max = max(z_minus_c)
    k = m + np.argmax(z_minus_c)
    
    # Validating if we stop or not the optimization
    if z_minus_c_max <= 0:   
        # Printing the final results
        print("\n\n\033[1mOptimality reached\033[0m")
        print("\nThe optimal BFS is")
        results_ls = [str(round(x_i, 4)) for x_i in x]
        print("\033[92m[ " + ', '.join(variables) + " ] = [ " + ', '.join(results_ls) + " ]\033[0m")
        perf_z = c.T @ x 
        print("\nWith performance z =", perf_z)
        
    else:   
        # Calculating and printing the y_ki's
        y_k = np.ravel(B_inv @ A[:,k])
        print("\nk =", k + 1, "-> column no.", k + 1, "of Matrix A (", variables[k], ")")
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
                    quot_ls.append(x[i] / y_k[i])
                    index_r_ls.append(i)
            quot_arr = np.array(quot_ls)
            index_r_arr = np.array(index_r_ls)
            r = index_r_arr[np.argmin(quot_arr)]
            x_k = min(quot_arr)
            print("\nr =", r + 1, "-> column no.", r + 1, "of Basis  (", variables[r], ")")
            print("x_k")
            print(x_k)

            # Exchanging the basic variable with index r and the not basic variable with index k, of the matrix A partitions
            var_aux = np.array(B[:, r])
            B[:, r] = N[:, k - m]
            N[:, k - m] = var_aux
            A = np.concatenate((B, N), axis = 1)
            
            # Exchanging the values located on indexes r and k of the vector c
            var_aux_c = c[r]
            c[r] = c[k]
            c[k] = var_aux_c
            
            # Exchanging the indexes
            print("\n\033[94m", variables[k], "enters\033[0m and \033[91m", variables[r], "leaves\033[0m the basis")
            aux_index = variables[r]
            variables[r] = variables[k]
            variables[k] = aux_index
            
            # Recursive call
            simplex(A, b, c, variables, itera + 1)

        else:
            
            print("\n\n\033[1mOptimization process stopped :(\033[0m")
            print("\nThe optimal BFS is not boundable")