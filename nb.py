#                           *****************************************************************
#                           * Naive-Bayes classifier to classify 0, 6 and 8 from mnist data *
#                           *****************************************************************

import numpy  as np
%matplotlib inline
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
(x,y),(x_val,y_val)=mnist.load_data()
a,b=[],[]
for i in range(len(x)):
  if y[i]==0 or y[i]==8 or y[i]==6:
    a.append(x[i])
    b.append(y[i])

a_flat=np.reshape(a,(np.shape(a)[0],28*28))
a_fin=a_flat/np.max(a_flat)

#preparing validation data
a_val,b_val=[],[]
for i in range(len(x_val)):
  if y_val[i]==0 or y_val[i]==8 or y_val[i]==6:
    a_val.append(x_val[i])
    b_val.append(y_val[i])

a_val_flat=np.reshape(a_val,(np.shape(a_val)[0],28*28))
a_val_fin=a_val_flat/np.max(a_val_flat)

import numpy as np
def naive_bayes(x,y,new_x,accu=0.01):                                 #     -------------------------------------------
 list_0=[0,6,8]                                                       #     |  Generalized defination of Naive-Bayes  |
 count_1,list_1=[],[]                                                 #     -------------------------------------------
 for iter_1 in range(len(list_0)):
  count_2=0
  list_2=[]
  for iter_2 in range(len(y)):
   if y[iter_2]==list_0[iter_1]:
    count_2+=1
    list_2.append(iter_2)
  count_1.append(count_2)
  list_1.append(list_2)
 probab=[i/len(y) for i in count_1]
 prob=[]
 for iter_4 in range(len(list_1)):
   prob_1=1
   for iter_5 in range(np.shape(x)[1]):
     count_3=0
     for iter_6 in list_1[iter_4]:
       if abs(x[iter_6][iter_5]-new_x[iter_5])==accu:
         count_3+=1
     const=count_3/count_1[iter_4]
     if const==0.0:
       continue
     else:
       prob_1=prob_1+const
   prob.append(prob_1)
 ans=[]
 for i in range(len(prob)):
   ans.append(prob[i]*probab[i])
 ans_1=ans/np.sum(ans)
 return list_0[np.argmax(ans_1)]

for i in range(40,100):
  print("test value is",b_val[i],"predicted value is",naive_bayes(a_fin,b,a_val_fin[i],0.0))
