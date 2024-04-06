#This function takes a vector of dependent variables (\textbf{y}) and a matrix of independent 
#variables as inputs. The outputs of this function are a vector of model coefficients, the vector of residuals 
#and the coefficient of determination

import numpy as np

def multi_regress(y, Z):
    # Get dimensions of Z
    n, m = Z.shape

    # Reshape y
    y = y.reshape(-1, 1)

    # Add an extra column to Z to handle constant
    Z = np.column_stack((np.ones(n), Z))

    # Setup matrix K and L
    K = np.zeros((m + 1, m + 1))
    L = np.zeros((m + 1, 1))

    for i in range(m + 1):
        for j in range(m + 1):
            K[i, j] = np.sum(Z[:, i] * Z[:, j])

    # First entry of K is number of dependent variables
    K[0, 0] = n

    # Setup L matrix
    for k in range(m + 1):
        L[k, 0] = np.sum(y[:,0] * Z[:, k])

    # Coefficients computed
    A =  np.linalg.inv(K) @ L
        
    # Error estimate
    e = y - Z @ A
    
    # Sum of squared
    S_r = np.sum(e**2)
    
    # difference between y and mean value of y (Simplest model)
    diff_y_ymean = (y - np.mean(y))**2
    

    S_t = np.sum(diff_y_ymean)

    # Coefficient of determination
    rsq = (S_t - S_r)/S_t

    return np.array(A) , np.array(e) , float(rsq)



    

    
    


    


