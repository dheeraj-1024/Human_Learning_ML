#                 **************************************************
#                 ***   PROGRAM FOR N-TH ORDER POLYNOMIAL FIT    ***
#                 **************************************************

import numpy as np
import matplotlib.pyplot as plt

x=np.array([i for i in range(1,100)])
y=np.array([5343*i**2+3.14786767 for i in x])

n=2                     # <-------------------|  degree to be fitted
A=np.array([[np.sum(x**(i+j)) for i in range(n,-1,-1)] for j in range(n,-1,-1)])
B=np.array([[np.sum(y*(x**j))] for j in range(n,-1,-1)])
w=np.matmul(np.linalg.inv(A),B)
print("weights in decreasing order are : \n",w)
plt.plot(x,y,"bo",label="raw data")
plt.plot(x,[np.sum([w[i]*(x[j]**(n-i)) for i in range(len(w))]) for j in range(len(x))],"go",label="fit data")
plt.legend()
plt.show()
