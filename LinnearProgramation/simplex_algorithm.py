import numpy as np

def get_data():
    print('Obteniendo datos de la matriz A')
    number_ecuations = int(input('Escribe el número de ecuaciones de restricción: '))
    A = list()
    for i in range(number_ecuations):
        row = list()
        for j in range(number_ecuations * 2):
            element = int(input(f'Escribe el número de la fila {i} columna {j}: '))
            row.append(element)
        A.append(row)
    A = np.array(A)
    print('Obteniendo datos del vector C')
    C = list()
    for i in range(number_ecuations * 2):
        element = int(input(f'Escribe el elemento {i}: '))
        C.append(element)
    C = np.array(C)
    print('Obteniendo datos del vector b')
    b = list()
    for i in range(number_ecuations):
        element = int(input(f'Escribe el elemento {i} de b: '))
        b.append(element)
    b = np.array(b)
    return [A, C, b]



def minimization():
    A, C, b = get_data()
    result = True
    while result == True:
        # Obtenemos B
        number_ecuations = len(A)
        B = list()
        for i in range(number_ecuations):
            row = list()
            for j in range(number_ecuations):
                row.append(A[i,j])
            B.append(row)
        B = np.array(B)
        print(B)
        # Obtenemos inversa de B
        B_ = np.linalg.inv(B)
        print(B_)
        # Obtenemos XB
        XB = B_.dot(b)
        print(XB)
        size_XB = len(XB)
        XN = list()
        for i in range(2 * size_XB):
            if i < size_XB:
                XN.append(XB[i])
            else:
                XN.append(0)
        print(XN)
        result = False