import numpy as np
from tabulate import tabulate

'''
x = 5
b = 2

A = x*b

terminal_output = "hello EAE 127 :)"
print(terminal_output)

'''

x=np.linspace(0,5,6)

#2D Arrays
X = np.meshgrid(x,x)
X=X[0]
print(X)
print('\n')

column = [(X[0])[0],(X[1])[0], (X[2])[0], (X[3])[0], (X[4])[0], (X[5])[0]]

def getColumn(two_array, column_num):
    amount_col = len(two_array)
    temp_column = np.zeros(amount_col)
    for i in range(0,6):
        temp_column[i] = two_array[i][column_num]
    return(temp_column)


showCol = [
    ['Column 1', getColumn(X,0)]
]

print(tabulate(showCol))